from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.worker import DistributedWorker
from tensorlink.ml.module import DistributedModel
from tensorlink.ml.utils import load_models_cache, save_models_cache
from tensorlink.api.models import GenerationRequest

from transformers import AutoTokenizer
from collections import defaultdict
import torch
import logging
import json
import time
import gc
import re
import os

# Path to package root
base_dir = os.path.dirname(os.path.abspath(__file__))

# Path to config/models.json relative to this script
SUPPORTED_MODELS_PATH = os.path.join(base_dir, "..", "config", "models.json")

with open(SUPPORTED_MODELS_PATH, "rb") as f:
    MODELS = json.load(f)
    DEFAULT_MODELS = MODELS["DEFAULT_MODELS"]


def extract_assistant_response(text: str, model_name: str = None) -> str:
    """
    Universal extractor that removes system/user/thought tags and returns
    the final human-readable assistant response.
    """

    # Remove reasoning or hidden thought blocks (e.g. <think>...</think>)
    text = re.sub(
        r"<\s*(think|reflection|thought|internal|analysis)\s*>.*?<\s*/\1\s*>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove common chat tags used by newer models
    text = re.sub(r"<\|im_start\|>\s*\w+\s*", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|assistant\|>", "", text)
    text = re.sub(r"<\|user\|>", "", text)
    text = re.sub(r"<\|system\|>", "", text)

    # Strip out any prefixes like "assistant:" or "Assistant:"
    text = re.sub(r"(?i)\bassistant\s*[:：]\s*", "", text)

    # Remove lingering system/user scaffolding
    text = re.sub(r"(?i)\b(system|user)\s*[:：]\s*", "", text)
    text = text.strip().replace("\r", "")

    # If multiple paragraphs, prefer the last coherent chunk
    # (models sometimes prepend hidden reasoning)
    if "\n\n" in text:
        parts = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 10]
        if parts:
            text = parts[-1]

    # Fallback: if text still empty, just return as-is (safe default)
    return text.strip() or "[No output produced]"


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
        self.model_cache = load_models_cache()
        self.models = {}  # job_id -> model instance
        self.model_state = (
            {}
        )  # job_id -> state ("initializing" | "distributing" | "ready")
        self.public_models = defaultdict(list)  # Model name -> list(job_id)

        self.tokenizers = {}
        self.GC_CHECK_INTERVAL = 1_000
        self.CHECK_COUNTER = 1

        # Track models that are in the process of being initialized (job_id)
        self.models_initializing = set()

        # Configuration
        self.TRACKING_DAYS = 7  # Track requests for past 7 days
        self.MIN_REQUESTS_THRESHOLD = 10  # Minimum requests to consider auto-loading
        self.MAX_AUTO_MODELS = 10  # Maximum models to auto-load

    def _ensure_model_entry(self, model_name: str):
        """Ensure a model has an entry in the cache with proper structure"""
        if model_name not in self.model_cache:
            self.model_cache[model_name] = {
                "distribution": None,
                "demand_metrics": {
                    "request_timestamps": [],
                    "total_requests": 0,
                    "last_accessed": None,
                },
            }

    def _record_request(self, model_name: str):
        """Record a request timestamp for a model in the JSON cache"""
        self._ensure_model_entry(model_name)

        current_time = time.time()

        # Add timestamp to the list
        self.model_cache[model_name]["demand_metrics"]["request_timestamps"].append(
            current_time
        )
        self.model_cache[model_name]["demand_metrics"]["total_requests"] += 1
        self.model_cache[model_name]["demand_metrics"]["last_accessed"] = current_time

        # Keep only recent timestamps to prevent unlimited growth
        cutoff_time = current_time - (self.TRACKING_DAYS * 24 * 3600)
        timestamps = self.model_cache[model_name]["demand_metrics"][
            "request_timestamps"
        ]
        self.model_cache[model_name]["demand_metrics"]["request_timestamps"] = [
            ts for ts in timestamps if ts >= cutoff_time
        ]

        # Save updated metrics
        save_models_cache(self.model_cache)

    def _get_recent_request_count(self, model_name: str, days: int = None) -> int:
        """Get number of requests for a model in the past X days from JSON cache"""
        if days is None:
            days = self.TRACKING_DAYS

        if model_name not in self.model_cache:
            return 0

        cutoff_time = time.time() - (days * 24 * 3600)
        timestamps = (
            self.model_cache[model_name]
            .get("demand_metrics", {})
            .get("request_timestamps", [])
        )

        return sum(1 for timestamp in timestamps if timestamp >= cutoff_time)

    def _cleanup_old_requests(self):
        """Remove request timestamps older than tracking period from all models in cache"""
        cutoff_time = time.time() - (self.TRACKING_DAYS * 24 * 3600)
        updated = False

        for model_name in list(self.model_cache.keys()):
            if "demand_metrics" in self.model_cache[model_name]:
                old_count = len(
                    self.model_cache[model_name]["demand_metrics"]["request_timestamps"]
                )

                # Filter out old timestamps
                self.model_cache[model_name]["demand_metrics"]["request_timestamps"] = [
                    ts
                    for ts in self.model_cache[model_name]["demand_metrics"][
                        "request_timestamps"
                    ]
                    if ts >= cutoff_time
                ]

                new_count = len(
                    self.model_cache[model_name]["demand_metrics"]["request_timestamps"]
                )

                if old_count != new_count:
                    updated = True

                # Remove entries with no recent activity
                if (
                    new_count == 0
                    and self.model_cache[model_name].get("distribution") is None
                ):
                    del self.model_cache[model_name]
                    updated = True

        if updated:
            save_models_cache(self.model_cache)

    def _get_popular_models(self) -> list:
        """Get list of models sorted by popularity from JSON cache"""
        model_popularity = []

        for model_name, model_data in self.model_cache.items():
            request_count = self._get_recent_request_count(model_name)
            if request_count >= self.MIN_REQUESTS_THRESHOLD:
                model_popularity.append((model_name, request_count))

        # Sort by request count (descending)
        model_popularity.sort(key=lambda x: x[1], reverse=True)
        return [model_name for model_name, _ in model_popularity]

    def _is_model_ready(self, job_id: str) -> bool:
        """Check if a model is ready for inference"""
        return self.model_state.get(job_id) == "ready"

    def _manage_auto_loaded_models(self):
        """Manage auto-loaded models based on popularity from JSON cache, falling back to DEFAULT_MODELS"""

        # Get popular models based on their request counts
        model_demands = {}
        for model_name in self.model_cache.keys():
            model_demands[model_name] = self._get_recent_request_count(model_name)

        # Add DEFAULT_MODELS with minimum demand to keep them warm
        for m in DEFAULT_MODELS:
            model_demands[m] = max(model_demands.get(m, 0), self.MIN_REQUESTS_THRESHOLD)

        if not model_demands:
            return

        total_requests = sum(max(v, 0) for v in model_demands.values())
        if total_requests == 0:
            return

        # Get number of desired model instances based on demand
        desired_instances = {}
        for model_name, count in model_demands.items():
            share = count / total_requests
            desired_instances[model_name] = round(share * self.MAX_AUTO_MODELS)

        can_allocate = True
        # Ensure each model has at least one instance
        for model_name, desired in desired_instances.items():
            if not can_allocate:
                break

            current_total = len(self.public_models.get(model_name, []))
            current_total += sum(
                1 if job_id in self.models_initializing else 0
                for job_id in self.public_models[model_name]
            )

            if current_total == 0 and desired > 0:
                self.send_request(
                    "debug_print",
                    (
                        f"Initializing first instance of {model_name}",
                        "cyan",
                        logging.INFO,
                    ),
                )
                can_allocate = self._initialize_hosted_job(model_name)

        # Finalize any first-load initializations
        if self.models_initializing:
            self._try_finalize_initializing_models()

        # Allocate duplicates based on proportional demand
        for model_name, target_count in desired_instances.items():
            if not can_allocate:
                break

            current_total = len(self.public_models.get(model_name, []))
            current_total += sum(
                1 if job_id in self.models_initializing else 0
                for job_id in self.public_models[model_name]
            )

            if current_total < target_count:
                to_launch = target_count - current_total
                for _ in range(to_launch):
                    self.send_request(
                        "debug_print",
                        (
                            f"Scaling UP (duplicate) {model_name}: +1 instance",
                            "green",
                            logging.INFO,
                        ),
                    )
                    can_allocate = self._initialize_hosted_job(model_name)

                    if not can_allocate:
                        break

        # Finalize any duplicate initializations
        if self.models_initializing:
            self._try_finalize_initializing_models()

    def inspect_model(self, model_name: str, job_data: dict, hosted=False) -> dict:
        """Inspect a model to determine network requirements and store distribution in JSON cache"""
        parser = ModelParser(verbose=True)
        model_name: str = job_data.get("model_name", model_name)

        # Get network worker information to assign modules
        workers = self.send_request("get_workers", None)

        batch_size = job_data.get("batch_size", None)

        if batch_size is None:
            if job_data.get("training", False):
                batch_size = 256
            else:
                batch_size = 1

        # Load HF model, create and save distribution
        distribution = parser.create_distributed_config(
            model_name,
            workers=workers,
            training=job_data.get("training", False),
            trusted=False,
            handle_layers=False,
            input_obfuscation=False,
            optimizer_type=job_data.get("optimizer_type"),
            host_load_small=True if hosted else False,
            host_max_depth=1,
            host_threshold_mb=20,
            max_offload_depth=3,
            batch_size=batch_size,
            max_seq_len=job_data.get("max_seq_len", 4096),
            model_type=job_data.get("model_type", "chat"),
        )

        job_data["distribution"] = distribution

        if (
            len(distribution["config"]) == 0
            or len(distribution["config"])
            > 5  # TODO This limit on number of distributions is not ideal
            or not distribution["success"]
        ):
            return {}

        # Store distribution in JSON cache
        self._ensure_model_entry(model_name)
        self.model_cache[model_name]["distribution"] = distribution
        save_models_cache(self.model_cache)

        self.send_request(
            "debug_print",
            (
                f"DistributedValidator -> Retrieved HF model: {job_data}",
                "bright_blue",
                logging.DEBUG,
            ),
        )

        gc.collect()  # Force garbage collection

        # Send out job request
        try:
            new_job_data = self.send_request("send_job_request", job_data)
            return new_job_data

        except Exception as e:
            print(str(e))

    def check_node(self):
        """Check for node requests/updates"""
        try:
            # When running on the public network, manage models automatically
            if not self.node.init_kwargs.get("endpoint", False):
                # Periodic cleanup and model management
                if self.CHECK_COUNTER % self.GC_CHECK_INTERVAL == 0:
                    # Clean up old request data
                    self._cleanup_old_requests()

                    # Manage autoloaded models based on popularity (or DEFAULT_MODELS fallback)
                    self._manage_auto_loaded_models()

                    # Check if jobs are still active
                    for job_id, model in self.models.items():
                        model_name = model.model_name
                        if self._is_model_ready(job_id):
                            is_active = self.send_request(
                                "check_job", (model_name, job_id)
                            )
                            if not is_active:
                                self._remove_hosted_job(job_id)

                    self.CHECK_COUNTER = 1

                if self.models_initializing:
                    # Only call model management if we have models actively initializing
                    self._try_finalize_initializing_models()

            # Get job data for inspection
            job_data = self.send_request("get_jobs", None)

            if isinstance(job_data, dict):
                model_name: str = job_data.get("model_name", "")

                if job_data.get("api"):
                    payment = job_data.get("payment", 0)
                    time_limit = job_data.get("time", 1800)
                    job_id = job_data.get("id")

                    # Check if this is a public job and there are already models of this type
                    self._initialize_hosted_job(
                        model_name,
                        job_data=job_data,
                        payment=payment,
                        time_limit=time_limit,
                    )

                    # Try to finalize if already initializing
                    if job_id in self.models_initializing:
                        self._finalize_hosted_job(model_name)

                else:
                    # If request via user node, begin the model reqs inspection for the job request
                    self.inspect_model(model_name, job_data, hosted=False)

            # Check for inference generate calls
            for job_id, distributed_model in self.models.items():
                if self._is_model_ready(job_id):
                    model_name = distributed_model.model_name
                    # TODO Distinguish private generate requests from public ones so we dont use the same model?
                    generate_request = self.send_request(
                        "update_api_request", (model_name, job_id)
                    )
                    if generate_request:
                        self._handle_generate_request(generate_request, job_id)

        except Exception as e:
            logging.error(f"Error checking for jobs: {str(e)}")

        self.CHECK_COUNTER += 1

    # def _handle_check_model_status(self, model_name: str):
    #     """Check the loading status of a model"""
    #     if model_name in self.models:
    #         if self._is_model_ready(model_name):
    #             # Model is fully loaded
    #             return {
    #                 "status": "loaded",
    #                 "message": f"Model {model_name} is loaded and ready",
    #             }
    #         else:
    #             # Model is in the process of loading
    #             return {
    #                 "status": "loading",
    #                 "message": f"Model {model_name} is currently loading",
    #             }
    #
    #     elif model_name in self.models_initializing:
    #         return {
    #             "status": "loading",
    #             "message": f"Model {model_name} initialization in progress",
    #         }
    #     else:
    #         return {
    #             "status": "not_loaded",
    #             "message": f"Model {model_name} is not loaded",
    #         }

    def _handle_generate_request(self, request: GenerationRequest, job_id: str):
        # Record the request for tracking
        self._record_request(request.hf_name)

        if not self._is_model_ready(job_id):
            request.output = (
                "Model is currently not available through the Tensorlink API."
            )
        else:
            distributed_model = self.models[job_id]

            tokenizer = self.tokenizers[request.hf_name]

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
                outputs = distributed_model.generate(
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

            # Many models echo the prompt, so remove it
            if generated_text.startswith(formatted_prompt):
                request.output = generated_text[len(formatted_prompt) :].strip()
            else:
                request.output = generated_text

        # Return the clean response
        self.send_request("update_api_request", (request,))

    def _try_finalize_initializing_models(self):
        """Attempt to finalize all models that are currently initializing."""
        for job_id in list(self.models_initializing):
            if self._finalize_hosted_job(job_id):
                self.send_request(
                    "debug_print",
                    (
                        f"Successfully finalized model: {job_id}",
                        "green",
                        logging.INFO,
                    ),
                )

    def _initialize_hosted_job(
        self,
        model_name: str,
        payment: int = 0,
        time_limit: int = None,
        job_data: dict = None,
    ):
        """Initialize a hosted job by creating the distributed model and submitting inspection request."""
        if not job_data:
            job_data = {}

        try:
            # Prepare job data for inspection
            defaults = {
                "author": None,
                "active": True,
                "hosted": True,
                "training": False,
                "payment": payment,
                "time": time_limit,
                "capacity": 0,
                "n_pipelines": 1,
                "dp_factor": 1,
                "distribution": {"model_name": model_name},
                "model_type": "chat",
                "n_workers": 0,
                "model_name": model_name,
                "seed_validators": [],
            }

            for k, v in defaults.items():
                job_data.setdefault(k, v)

            # Inspect model to determine network requirements
            job_data = self.inspect_model(model_name, job_data, hosted=True)

            if not job_data:
                return False

            job_id = job_data.get("id")

            # Create distributed model instance
            distributed_model = DistributedModel(
                model_name,
                node=self.node,
                training=False,
            )

            self.models[job_id] = distributed_model

            if job_data.get("public"):
                self.public_models[model_name].append(job_id)

            self.model_state[job_id] = "initializing"
            self.models_initializing.add(job_id)
            return True

        except Exception as e:
            logging.error(f"Error initializing hosted job for {model_name}: {str(e)}")
            job_id = job_data.get("id")
            self.models_initializing.discard(job_id)
            del self.models[job_id]
            if job_id in self.model_state:
                del self.model_state[job_id]

            return False

    def _finalize_hosted_job(self, job_id: str):
        """Finalize a hosted job by setting up the distributed model with workers."""
        try:
            # Check if we have module info ready
            args = self.send_request("check_module", job_id)

            if not args or not isinstance(args, dict):
                # Module not ready yet
                return False

            model_name = args["model_name"]
            distribution = args["distribution"]
            optimizer_name = args["optimizer"]
            training = args["training"]

            # Check if model is in initialization state
            if job_id not in self.models:
                return False

            # Get the DistributedModel instance
            distributed_model = self.models[job_id]

            # Update state
            self.model_state[job_id] = "distributing"

            # Register the distributed model's modules
            for module_id, module_info in distribution.items():
                module_info["job_id"] = job_id
                self.modules[module_id] = module_info

            # Distribute the model across workers
            distributed_model.distribute_model(distribution)
            distributed_model.job_id = job_id

            # Load tokenizer
            if model_name not in self.tokenizers:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

            # Mark as ready
            self.model_state[job_id] = "ready"
            self.models_initializing.discard(job_id)

            self.send_request(
                "debug_print",
                (
                    f"Finalized hosted job for {model_name} with module_id {module_id}",
                    "green",
                    logging.INFO,
                ),
            )

            return True

        except Exception as e:
            logging.error(f"Error finalizing hosted job for {model_name}: {str(e)}")
            self.models_initializing.discard(job_id)
            if job_id in self.models:
                del self.models[job_id]
            return False

    def _remove_hosted_job(self, job_id: str):
        """Remove a hosted job and clean up all associated resources"""
        try:
            # Remove from initializing set if present
            self.models_initializing.discard(job_id)

            distributed_model = self.models[job_id]
            model_name = distributed_model.model_name

            # Clean up tokenizer if no other models require it
            if (
                model_name in self.tokenizers
                and len(self.public_models[model_name]) <= 1
            ):
                del self.tokenizers[model_name]
                self.send_request(
                    "debug_print",
                    (f"Removed tokenizer for {model_name}", "yellow", logging.INFO),
                )

            if model_name in self.public_models:
                if job_id in self.public_models[model_name]:
                    self.public_models[model_name].remove(job_id)

            # Clean up state tracking
            if job_id in self.model_state:
                del self.model_state[job_id]

            # Clean up model reference
            del self.models[job_id]

            # Find and remove any module entries that reference this model
            modules_to_remove = []
            for module_id, module_data in self.modules.items():
                # Check if this module belongs to the model we're removing
                if module_data.get("name") == model_name:
                    if module_data.get("job_id") == job_id:
                        modules_to_remove.append(module_id)

            for module_id in modules_to_remove:
                del self.modules[module_id]
                self.send_request(
                    "debug_print",
                    (
                        f"Removed module reference {module_id} for {model_name}",
                        "yellow",
                        logging.INFO,
                    ),
                )

            # Only remove model cache if it has no distribution data and no recent requests
            if (
                model_name in self.model_cache
                and self.model_cache[model_name].get("distribution") is not None
                and self._get_recent_request_count(model_name, days=1) == 0
            ):

                # Keep demand metrics but clear distribution if no recent activity
                self.model_cache[model_name]["distribution"] = None
                save_models_cache(self.model_cache)

                self.send_request(
                    "debug_print",
                    (
                        f"Cleared distribution cache for {model_name}",
                        "yellow",
                        logging.INFO,
                    ),
                )

            # Send cleanup request to node
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
        time.sleep(0.001)
