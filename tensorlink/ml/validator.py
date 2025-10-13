from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.worker import DistributedWorker
from tensorlink.ml.module import DistributedModel
from tensorlink.ml.utils import load_models_cache, save_models_cache
from tensorlink.api.node import GenerationRequest

from transformers import AutoTokenizer
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
        self.models = {}
        self.tokenizers = {}
        self.GC_CHECK_INTERVAL = 1000
        self.CHECK_COUNTER = 1

        # Track models that are in the process of being initialized
        self.models_initializing = set()

        # Configuration
        self.TRACKING_DAYS = 7  # Track requests for past 7 days
        self.MIN_REQUESTS_THRESHOLD = 10  # Minimum requests to consider auto-loading
        self.MAX_AUTO_MODELS = 5  # Maximum models to auto-load

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

    def _manage_auto_loaded_models(self):
        """Manage auto-loaded models based on popularity from JSON cache, falling back to DEFAULT_MODELS"""
        popular_models = self._get_popular_models()

        # If no popular models tracked yet, use DEFAULT_MODELS as fallback
        models_to_load = DEFAULT_MODELS[: self.MAX_AUTO_MODELS]
        # if not popular_models:
        #     models_to_load = DEFAULT_MODELS[: self.MAX_AUTO_MODELS]
        # else:
        #     models_to_load = popular_models[: self.MAX_AUTO_MODELS]
        #     self.send_request(
        #         "debug_print",
        #         (f"Loading popular models: {models_to_load}", "blue", logging.INFO),
        #     )

        # Load models up to the limit
        for model_name in models_to_load:
            if (
                model_name not in self.models
                and model_name not in self.models_initializing
            ):
                self.send_request(
                    "debug_print",
                    (f"Auto-loading model: {model_name}", "green", logging.INFO),
                )
                self.models_initializing.add(model_name)
                self._initialize_hosted_job(model_name)

        # Continue initialization for models that are in progress
        for model_name in list(self.models_initializing):
            if model_name in models_to_load:  # Still wanted
                # Try second initialization call
                self._initialize_hosted_job(model_name)
                # Check if initialization is complete
                if model_name in self.models and isinstance(
                    self.models[model_name], str
                ):
                    # Model is fully initialized (module_id is now a string)
                    self.models_initializing.discard(model_name)
                    self.send_request(
                        "debug_print",
                        (
                            f"Completed auto-loading model: {model_name}",
                            "green",
                            logging.INFO,
                        ),
                    )
            else:
                # Model no longer wanted, cancel initialization
                self.models_initializing.discard(model_name)
                if model_name in self.models:
                    self._remove_hosted_job(model_name)

        # Remove models not in the current priority list
        currently_loaded = [
            name for name in self.models.keys() if isinstance(self.models[name], str)
        ]
        for model_name in currently_loaded:
            if model_name not in models_to_load:
                recent_requests = self._get_recent_request_count(
                    model_name, days=1
                )  # Check last day
                if recent_requests < 5:  # Low recent activity
                    self.send_request(
                        "debug_print",
                        (
                            f"Removing unpopular model: {model_name}",
                            "yellow",
                            logging.INFO,
                        ),
                    )
                    self._remove_hosted_job(model_name)

    def inspect_model(self, model_name: str, job_data: dict = None):
        """Inspect a model to determine network requirements and store distribution in JSON cache"""
        parser = ModelParser()
        model_name = job_data.get("model_name", model_name)

        # Load HF model, create and save distribution
        distribution = parser.create_distributed_config(
            model_name,
            training=job_data.get("training", False),
            trusted=False,
        )
        job_data["distribution"] = distribution

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
            self.send_request("send_job_request", job_data)
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
                    for model_name in list(self.models.keys()):
                        if isinstance(
                            self.models[model_name], str
                        ):  # Only check fully loaded models
                            is_active = self.send_request("check_job", (model_name,))
                            if not is_active:
                                self._remove_hosted_job(model_name)

                    self.CHECK_COUNTER = 1

                elif self.models_initializing:
                    # Only call model management if we have models actively initializing
                    self._manage_auto_loaded_models()

            # Get job data for inspection
            job_data = self.send_request("get_jobs", None)

            if isinstance(job_data, dict):
                # Offload model inspection to a background thread to avoid blocking
                self.inspect_model(job_data.get("model_name"), job_data)

            # Check for inference generate calls
            for model_name, module_id in self.models.items():
                if (
                    isinstance(module_id, str) and module_id in self.modules
                ):  # Only process fully loaded models
                    generate_request = self.send_request(
                        "update_api_request", (model_name, module_id)
                    )
                    if generate_request:
                        self._handle_generate_request(generate_request)

        except Exception as e:
            logging.error(f"Error checking for jobs: {str(e)}")

        self.CHECK_COUNTER += 1

    def _handle_check_model_status(self, model_name: str):
        """Check the loading status of a model"""
        if model_name in self.models:
            module_id = self.models[model_name]
            if isinstance(module_id, str):
                # Model is fully loaded
                return {
                    "status": "loaded",
                    "message": f"Model {model_name} is loaded and ready",
                    "module_id": module_id,
                }
            else:
                # Model is in the process of loading
                return {
                    "status": "loading",
                    "message": f"Model {model_name} is currently loading",
                }
        elif model_name in self.models_initializing:
            return {
                "status": "loading",
                "message": f"Model {model_name} initialization in progress",
            }
        else:
            return {
                "status": "not_loaded",
                "message": f"Model {model_name} is not loaded",
            }

    def _handle_load_model(self, model_name: str):
        """Handle explicit request to load a model"""
        try:
            # Check if already loaded or loading
            if model_name in self.models or model_name in self.models_initializing:
                self.send_request(
                    "debug_print",
                    (
                        f"Model {model_name} is already loaded or loading",
                        "yellow",
                        logging.INFO,
                    ),
                )
                return

            # Add to initializing set
            self.models_initializing.add(model_name)

            self.send_request(
                "debug_print",
                (f"Loading model on demand: {model_name}", "green", logging.INFO),
            )

            # Initialize the model
            self._initialize_hosted_job(model_name)

        except Exception as e:
            logging.error(f"Error loading model {model_name}: {str(e)}")
            self.models_initializing.discard(model_name)

    def _handle_generate_request(self, request: GenerationRequest):
        # Record the request for tracking
        self._record_request(request.hf_name)

        if request.hf_name not in self.models or not isinstance(
            self.models[request.hf_name], str
        ):
            request.output = (
                "Model is currently not available through the Tensorlink API."
            )
        else:
            module_id = self.models[request.hf_name]
            model = self.modules[module_id]
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

        # Check if the model loading is complete across workers and ready to go (second call)
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
                self.modules[module_id].distribute_model(distribution)
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

        # If not, check if we can spin up the model (first call)
        else:
            # Small init sleep time
            if model_name in self.models:
                time.sleep(20)

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
            # Remove from initializing set if present
            self.models_initializing.discard(model_name)

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
                self.send_request(
                    "debug_print",
                    (
                        f"Removed model reference for {model_name}",
                        "yellow",
                        logging.INFO,
                    ),
                )

            # Clean up module if it exists
            if isinstance(module_id, str) and module_id in self.modules:
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

            # Only remove if it has no distribution data and no recent requests
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
        time.sleep(0.005)
