from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.utils import estimate_hf_model_memory, get_hf_model
from tensorlink.ml.worker import DistributedWorker
from tensorlink.ml.module import DistributedModel
from tensorlink.api.node import GenerationRequest

from collections import defaultdict
from transformers import AutoTokenizer
import heapq
import torch
import logging
import json
import time
import gc
import re
import os


MODELS_CACHE_PATH = "logs/models.json"

# Path to package root (where this file lives)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Path to config/models.json relative to this script
SUPPORTED_MODELS_PATH = os.path.join(base_dir, "..", "config", "models.json")

with open(SUPPORTED_MODELS_PATH, "rb") as f:
    MODELS = json.load(f)
    FREE_MODELS = MODELS["FREE_MODELS"]


def load_models():
    try:
        with open(MODELS_CACHE_PATH, "r") as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_models(models):
    with open(MODELS_CACHE_PATH, "w") as f:
        json.dump(models, f, indent=4)


def extract_assistant_response(text: str, model_name: str = None) -> str:
    # Split on 'assistant' prompts
    assistant_responses = re.split(r"\bassistant\b", text)
    if len(assistant_responses) < 2:
        return text.strip()

    # Take the last assistant response and strip off any trailing user/system prompts
    last_response = assistant_responses[-1].strip()

    # Optionally remove any 'user' or 'system' that follows
    last_response = re.split(r"\b(user|system)\b", last_response)[0].strip()

    return last_response


def format_chat_prompt(model_name, current_message, history):
    """Format the chat history and current message into a prompt suitable for the specified model."""

    # Different models require different formatting
    if "Qwen" in model_name:
        # Qwen-specific formatting
        system_prompt = (
            "You are a helpful assistant. Respond directly to the user's questions."
        )

        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        # Add conversation history
        if history and len(history) > 0:
            for msg in history:
                role = msg["role"]
                content = msg["content"]
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        # Add the current message
        formatted_prompt += f"<|im_start|>user\n{current_message}<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"

        return formatted_prompt

    elif "llama" in model_name.lower():
        # Llama-style formatting
        system_prompt = (
            "You are a helpful assistant. Respond directly to the user's questions."
        )
        formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

        # Add conversation history
        if history and len(history) > 0:
            for i, msg in enumerate(history):
                if msg["role"] == "user":
                    if i > 0:
                        formatted_prompt += "[/INST]\n\n[INST] "
                    formatted_prompt += f"{msg['content']}"
                else:  # assistant
                    formatted_prompt += f" [/INST]\n\n{msg['content']}\n\n[INST] "

        # Add the current message and prepare for response
        formatted_prompt += f"{current_message} [/INST]\n\n"

        return formatted_prompt

    else:
        # Generic formatting for other models
        system_prompt = (
            "You are a helpful assistant. Respond directly to the user's questions."
        )
        formatted_prompt = f"System: {system_prompt}\n\n"

        # Add conversation history
        if history and len(history) > 0:
            for msg in history:
                role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
                formatted_prompt += f"{role_prefix}{msg['content']}\n\n"

        # Add the current message
        formatted_prompt += f"User: {current_message}\n\nAssistant: "

        return formatted_prompt


class DistributedValidator(DistributedWorker):
    def __init__(self, node, trusted=False):
        super().__init__(node, trusted)
        self.model_cache = load_models()
        self.models = {}
        self.tokenizers = {}
        self.models_initialized = 0
        self.GC_CHECK_INTERVAL = 1000
        self.CHECK_COUNTER = 1

        self.model_demand_history = defaultdict(list)
        self.loading_queue = []
        self.model_load_timestamps = {}
        self.demand_check_interval = 50
        self.max_concurrent_models = 3
        self.min_demand_threshold = 1
        self.demand_window_size = 100  # Number of recent demand samples to consider
        self.pending_requests = defaultdict(list)  # Queue requests waiting for models
        self.preload_candidates = set()  # Models to preload based on patterns
        self.request_urgency_threshold = 3  # Auto-load if this many requests queue up

    def inspect_model(self, model_name: str, job_data: dict = None):
        """Inspect a model to determine network requirements (ie vram, n modules, etc)"""
        parser = ModelParser()
        model_name = job_data.get("model_name", model_name)

        if job_data.get("vram", 0) < 24e9:
            # Load HF model, create and save distribution
            # model, tokenizer = get_hf_model(model_name, tokenizer=True)
            distribution = parser.create_distributed_config(
                model_name,
                training=job_data.get("training", False),
                trusted=False,
            )
            job_data["distribution"] = distribution
            self.model_cache[model_name] = {"distribution": distribution}

            self.send_request(
                "debug_print",
                (
                    f"DistributedValidator -> Retrieved HF model: {job_data}",
                    "bright_blue",
                    logging.DEBUG,
                ),
            )

            # del model
            # del tokenizer
            gc.collect()  # Force garbage collection

            # Send out job request
            try:
                self.send_request("send_job_request", job_data)
            except Exception as e:
                print(str(e))

    def model_manager(self):
        """Enhanced model management with urgency handling"""
        try:
            # Get real-time demand from API layer
            current_api_demand = self.send_request("get_api_demand_stats", None) or {}

            # Merge with coordinator demand stats
            coordinator_demand = self.send_request("get_model_demand_stats", None) or {}

            # Combine demand sources (API requests + coordinator stats)
            total_demand = defaultdict(int)
            for source in [current_api_demand, coordinator_demand]:
                for model, count in source.items():
                    total_demand[model] += count

            self.update_demand_history(total_demand)

            # URGENT: Handle queued requests that need immediate model loading
            self.handle_urgent_requests()

            # Regular demand-based loading
            self.update_loading_queue()

            # Try to load highest priority model if we have capacity
            if self.loading_queue and len(self.models) < self.max_concurrent_models:
                model_name, priority = self.loading_queue[0]

                # Check if this is an urgent request
                is_urgent = (
                    len(self.pending_requests[model_name])
                    >= self.request_urgency_threshold
                )

                self.send_request(
                    "debug_print",
                    (
                        f"Loading model: {model_name} (priority: {priority:.1f}, urgent: {is_urgent})",
                        "green" if not is_urgent else "red",
                        logging.INFO,
                    ),
                )

                self.loading_queue.pop(0)
                self._initialize_hosted_job(model_name)
                self.model_load_timestamps[model_name] = time.time()

            # Unload low-demand models
            self.manage_model_unloading()

        except Exception as e:
            logging.error(f"Error in enhanced model management: {str(e)}")

    def manage_model_unloading(self):
        """Smart model unloading with better heuristics"""
        if len(self.models) <= 1:
            return  # Keep at least one model

        models_to_unload = []
        for model_name in list(self.models.keys()):
            if self.should_unload_model(model_name):
                # Calculate unload score (higher = more likely to unload)
                time_since_use = time.time() - self.model_load_timestamps.get(
                    model_name, time.time()
                )
                recent_demand = sum(
                    count for ts, count in self.model_demand_history[model_name][-3:]
                )

                unload_score = time_since_use / (
                    recent_demand + 1
                )  # Avoid division by zero
                models_to_unload.append((model_name, unload_score))

        if models_to_unload:
            # Sort by unload score and unload the highest scoring model
            models_to_unload.sort(key=lambda x: x[1], reverse=True)
            model_to_unload = models_to_unload[0][0]

            self.send_request(
                "debug_print",
                (
                    f"Smart unload: {model_to_unload} (score: {models_to_unload[0][1]:.1f})",
                    "yellow",
                    logging.INFO,
                ),
            )
            self._remove_hosted_job(model_to_unload)

    def handle_urgent_requests(self):
        """Handle requests that have been waiting for unavailable models"""
        for model_name, requests in self.pending_requests.items():
            if (
                len(requests) >= self.request_urgency_threshold
                and model_name not in self.models
            ):
                # Force-load model for urgent requests
                if len(self.models) >= self.max_concurrent_models:
                    # Free up space by unloading least recently used model
                    lru_model = min(
                        self.model_load_timestamps.keys(),
                        key=lambda m: self.model_load_timestamps[m],
                    )
                    if lru_model != model_name:  # Don't unload what we're about to load
                        self.send_request(
                            "debug_print",
                            (
                                f"Emergency unload of {lru_model} for urgent {model_name}",
                                "yellow",
                                logging.WARNING,
                            ),
                        )
                        self._remove_hosted_job(lru_model)

                # Prioritize this model for immediate loading
                self.loading_queue.insert(0, (model_name, 999))  # Max priority

    def update_loading_queue(self):
        """Update the priority queue of models to load based on current demand"""
        current_demand = self.get_model_demand_stats()
        self.update_demand_history(current_demand)

        # Clear existing queue
        self.loading_queue.clear()

        # Calculate priorities for all free models
        model_priorities = []
        for model_name in FREE_MODELS:
            if model_name not in self.models:  # Only queue unloaded models
                priority = self.calculate_model_priority(model_name)
                if priority >= self.min_demand_threshold:
                    # Use negative priority for max-heap behavior
                    heapq.heappush(model_priorities, (-priority, model_name))

        # Convert to loading queue (keep top models within resource limits)
        available_slots = self.max_concurrent_models - len(self.models)
        while model_priorities and len(self.loading_queue) < available_slots:
            neg_priority, model_name = heapq.heappop(model_priorities)
            self.loading_queue.append((model_name, -neg_priority))

        # Sort by priority (highest first)
        self.loading_queue.sort(key=lambda x: x[1], reverse=True)

        if self.loading_queue:
            self.send_request(
                "debug_print",
                (
                    f"Updated loading queue: {[(name, f'{priority:.1f}') for name, priority in self.loading_queue]}",
                    "cyan",
                    logging.INFO,
                ),
            )

    def should_unload_model(self, model_name):
        """Determine if a model should be unloaded due to low demand or expiration"""
        if len(self.pending_requests[model_name]) > 0:
            return False

        if model_name not in FREE_MODELS or model_name not in self.models:
            return False

        # Don't unload if recently loaded (give it time to be used)
        if model_name in self.model_load_timestamps:
            time_since_load = time.time() - self.model_load_timestamps[model_name]
            if time_since_load < 600:  # 10 minutes minimum runtime
                return False

        # Check recent demand
        if model_name in self.model_demand_history:
            recent_history = self.model_demand_history[model_name][
                -3:
            ]  # Last 3 samples
            if recent_history:
                recent_demand = sum(count for ts, count in recent_history)
                if recent_demand < 2:  # Very low recent demand
                    return True

        return False

    def calculate_model_priority(self, model_name):
        """Calculate priority score for loading a model based on demand patterns"""
        if model_name not in self.model_demand_history:
            return 0

        history = self.model_demand_history[model_name]
        if len(history) < 2:
            return 0

        # Recent demand (higher weight for recent requests)
        recent_demand = sum(count for ts, count in history[-self.demand_window_size :])

        # Demand trend (is it increasing?)
        if len(history) >= 4:
            recent_avg = sum(count for ts, count in history[-2:]) / 2
            older_avg = sum(count for ts, count in history[-4:-2]) / 2
            trend_multiplier = max(1.0, recent_avg / max(older_avg, 1))
        else:
            trend_multiplier = 1.0

        # Penalty if model is already loaded
        load_penalty = 0.1 if model_name in self.models else 1.0

        # Priority score (higher = more priority)
        priority = recent_demand * trend_multiplier * load_penalty

        return priority

    def update_demand_history(self, current_demand):
        """Update rolling window of demand history for each model"""
        current_time = time.time()

        for model_name, request_count in current_demand.items():
            if model_name in FREE_MODELS:
                # Add current demand sample with timestamp
                self.model_demand_history[model_name].append(
                    (current_time, request_count)
                )

                # Keep only recent samples within the window
                cutoff_time = current_time - 300  # 5 minutes window
                self.model_demand_history[model_name] = [
                    (ts, count)
                    for ts, count in self.model_demand_history[model_name]
                    if ts > cutoff_time
                ]

    def get_model_demand_stats(self):
        """Get hashmap of model_names to num generate requests from the coordinator"""
        try:
            # This would be your API call to get current demand stats
            demand_stats = self.send_request("get_model_demand_stats", None)
            if isinstance(demand_stats, dict):
                return demand_stats
            return {}
        except Exception as e:
            logging.error(f"Error getting demand stats: {str(e)}")
            return {}

    def check_node(self):
        """Check for node requests/updates"""
        try:
            # When running on the public network, we maintain an active set of self-hosted API-accessible models
            if not self.node.init_kwargs.get("local_test", False):
                if self.CHECK_COUNTER % self.demand_check_interval == 0:
                    self.model_manager()

                # Check if job is still active
                if self.CHECK_COUNTER % self.GC_CHECK_INTERVAL == 0:
                    for model_name in self.models:
                        if model_name in FREE_MODELS:
                            is_active = self.send_request("check_job", (model_name,))
                            if not is_active:
                                self._remove_hosted_job(model_name)

                    self.CHECK_COUNTER = 1

            # Get job data for inspection
            job_data = self.send_request("get_jobs", None)

            if isinstance(job_data, dict):
                # Offload model inspection to a background thread to avoid blocking
                self.inspect_model(job_data.get("model_name"), job_data)

            # Check for inference generate calls
            for model_name, module_id in self.models.items():
                if module_id in self.modules:
                    generate_request = self.send_request(
                        "update_api_request", (model_name, module_id)
                    )
                    if generate_request:
                        self._handle_generate_request(generate_request)

        except Exception as e:
            logging.error(f"Error checking for jobs: {str(e)}")

        self.CHECK_COUNTER += 1

    def initialize_hosted_jobs(self):
        """Initialize hosted jobs with proper thread coordination"""
        # Use thread pool to avoid blocking the main thread
        for model_name in FREE_MODELS:
            self._initialize_hosted_job(model_name)

    def _handle_generate_request(self, request: GenerationRequest):
        if request.hf_name not in self.models:
            # Queue the request and trigger urgent loading if needed
            self.pending_requests[request.hf_name].append(request)

            # Trigger model loading if urgency threshold reached
            if (
                len(self.pending_requests[request.hf_name])
                >= self.request_urgency_threshold
            ):
                self.send_request(
                    "debug_print",
                    (
                        f"Urgent loading triggered for {request.hf_name}",
                        "red",
                        logging.WARNING,
                    ),
                )

            request.output = f"Model {request.hf_name} is loading due to demand. Please retry in 30-60 seconds."

        else:
            module_id = self.models[request.hf_name]
            model = self.modules[module_id]
            tokenizer = self.tokenizers[request.hf_name]

            # Remove from pending queue if it was there
            if request in self.pending_requests[request.hf_name]:
                self.pending_requests[request.hf_name].remove(request)

            # Format chat history into a standardized prompt
            formatted_prompt = format_chat_prompt(
                request.hf_name, request.message, request.history
            )

            # Tokenize formatted prompt
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=request.max_length if request.max_length else 512,
            )

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=(
                        request.max_new_tokens
                        if hasattr(request, 'max_new_tokens')
                        else 2048
                    ),
                    temperature=request.temperature if request.temperature else 0.6,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=(
                        request.do_sample if hasattr(request, 'do_sample') else True
                    ),
                    num_beams=request.num_beams if request.num_beams else 1,
                )

            # Decode generated tokens
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response from the generated text
            clean_response = extract_assistant_response(generated_text, request.hf_name)
            request.output = clean_response

        # Return the clean response
        self.send_request("update_api_request", (request,))

    def _initialize_hosted_job(self, model_name: str):
        """Method that can be invoked twice, once to begin setup of the job, and a second
        time to finalize the job init."""
        args = self.send_request("check_module", None)

        # Check if the model loading is complete across workers and ready to go
        if model_name in self.models and args:
            if isinstance(args, tuple):
                (
                    file_name,
                    module_id,
                    distribution,
                    module_name,
                    optimizer_name,
                    training,
                ) = args
                self.modules[module_id] = self.models.pop(model_name)
                self.models[model_name] = module_id
                self.models_initialized += 1
                self.modules[module_id].distribute_model(distribution)
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

        # If not, check if we can spin up the model
        else:
            if model_name in self.models:
                time.sleep(30)

            distributed_model = DistributedModel(model_name, node=self.node)
            self.models[model_name] = distributed_model
            job_data = {
                "author": None,
                "active": True,
                "hosted": True,
                "training": False,
                "payment": 0,
                "capacity": 0,
                "n_pipelines": 1,
                "dp_factor": 1,
                "distribution": {"model_name": model_name},
                "n_workers": 0,
                "model_name": model_name,
                "seed_validators": [],
            }
            self.inspect_model(model_name, job_data)

    def _remove_hosted_job(self, model_name: str):
        """Remove a hosted job and clean up all associated resources"""
        try:
            # Get the module_id if the model is tracked
            module_id = None
            if model_name in self.models:
                module_id = self.models[model_name]

            # Clean up tokenizer
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
                self.send_request(
                    "debug_print",
                    (f"Removed tokenizer for {model_name}", "yellow", logging.INFO),
                )

            # Clean up model reference
            if model_name in self.models:
                del self.models[model_name]
                self.models_initialized = max(0, self.models_initialized - 1)
                self.send_request(
                    "debug_print",
                    (
                        f"Removed model reference for {model_name}",
                        "yellow",
                        logging.INFO,
                    ),
                )

            # Clean up module if it exists
            if module_id and module_id in self.modules:
                # Notify the module to clean up its distributed components
                try:
                    if hasattr(self.modules[module_id], 'cleanup_distributed_model'):
                        self.modules[module_id].cleanup_distributed_model()
                except Exception as e:
                    logging.warning(
                        f"Error during distributed model cleanup for {model_name}: {str(e)}"
                    )

                del self.modules[module_id]
                self.send_request(
                    "debug_print",
                    (
                        f"Removed module {module_id} for {model_name}",
                        "yellow",
                        logging.INFO,
                    ),
                )

            # Clean up model cache
            if model_name in self.model_cache:
                del self.model_cache[model_name]
                self.send_request(
                    "debug_print",
                    (f"Removed cache entry for {model_name}", "yellow", logging.INFO),
                )

            # Send cleanup request to coordinator/scheduler
            try:
                self.send_request("remove_job", {"model_name": model_name})
            except Exception as e:
                logging.warning(
                    f"Error sending job removal request for {model_name}: {str(e)}"
                )

            # Force garbage collection to free memory
            gc.collect()

            self.send_request(
                "debug_print",
                (
                    f"Successfully removed hosted job: {model_name}",
                    "green",
                    logging.INFO,
                ),
            )

        except Exception as e:
            logging.error(f"Error removing hosted job {model_name}: {str(e)}")
            self.send_request(
                "debug_print",
                (
                    f"Failed to remove hosted job {model_name}: {str(e)}",
                    "red",
                    logging.ERROR,
                ),
            )

    def main_loop(self):
        self.check_node()
        time.sleep(0.005)
