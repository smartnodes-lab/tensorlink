from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.utils import estimate_hf_model_memory, get_hf_model
from tensorlink.ml.worker import DistributedWorker
from tensorlink.ml.module import DistributedModel
from tensorlink.api.node import GenerationRequest

from transformers import AutoTokenizer
import torch
import logging
import json
import time
import gc
import re


MODELS_CACHE_PATH = "logs/models.json"
SUPPORTED_MODELS_PATH = "tensorlink/config/models.json"


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

    def inspect_model(self, model_name: str, job_data: dict = None):
        """Inspect a model with proper coordination with other threads"""
        parser = ModelParser()
        model_name = job_data.get("model_name", model_name)

        if job_data.get("vram", 0) < 24e9:
            # Load HF model, create and save distribution
            # model, tokenizer = get_hf_model(model_name, tokenizer=True)
            distribution = parser.create_distributed_config(
                model_name,  # model,
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

    def check_node(self):
        """Check for node updates with efficient scheduling and locking and perform job inspection"""
        try:
            if not self.node.init_kwargs.get("local_test", False):
                if self.models_initialized < len(FREE_MODELS):
                    time.sleep(10)  # Temporary sleep to allow network bootstrap first
                    self.initialize_hosted_jobs()

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

    def initialize_hosted_jobs(self):
        """Initialize hosted jobs with proper thread coordination"""
        # Use thread pool to avoid blocking the main thread
        for model_name in FREE_MODELS:
            self._initialize_hosted_job(model_name)

    def _handle_generate_request(self, request: GenerationRequest):
        if request.hf_name in self.models:
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

            # Return the clean response
            self.send_request(
                "update_api_request", (request.hf_name, module_id, clean_response)
            )

    def _initialize_hosted_job(self, model_name: str):
        # Check if the model loading is complete across workers and ready to go
        args = self.send_request("check_module", None)
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

    def main_loop(self):
        self.check_node()
        time.sleep(0.005)
