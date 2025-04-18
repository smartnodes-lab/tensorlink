from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.utils import estimate_hf_model_memory, get_hf_model
from tensorlink.ml.worker import DistributedWorker

import threading
import logging
import json
import time
import gc


MODELS_PATH = "logs/models.json"

POPULAR_MODELS = []
FREE_MODELS = ["Qwen/Qwen2.5-7B-Instruct"]


def load_models():
    try:
        with open(MODELS_PATH, "r") as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_models(models):
    with open(MODELS_PATH, "w") as f:
        json.dump(models, f, indent=4)


class DistributedValidator(DistributedWorker):
    def __init__(self, node_requests, node_responses, mpc_lock, trusted=False):
        super().__init__(node_requests, node_responses, mpc_lock, trusted)
        self.models = load_models()
        self.hosted_models = {}

    def inspect_model(self, model_name: str, job_data: dict = None):
        # We want to load a model and check if its worthy of a job
        if job_data is None:
            job_data = {"hosted": True, "model_name": model_name}

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

            # Save the distribution
            self.models[model_name] = {"distribution": distribution}

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

        # Check if we already have the distribution saved
        # if (
        #     model_name in self.models
        #     and "distribution" in self.models[model_name]
        # ):
        #     # Use the saved distribution
        #     distribution = self.models[model_name]["distribution"]
        #
        #     # Update RAM and VRAM estimates from saved data
        #     job_data["vram"] = sum(
        #         [v["size"] for v in distribution.values()]
        #     )
        #     job_data["ram"] = sum(
        #         [v["size"] for v in distribution.values()]
        #     )
        #
        #     self.send_request(
        #         "debug_print",
        #         (
        #             f"DistributedValidator -> Retrieved cached HF model: {job_data}",
        #             "bright_blue",
        #             logging.INFO,
        #         ),
        #     )
        # else:

    def check_node(self):
        """Check for node updates with efficient scheduling and locking and perform job inspection"""
        # Skip if ML operation is in progress to avoid contention
        if self.ml_operation_in_progress.is_set():
            return

        # Try to acquire semaphore (non-blocking)
        if not self.node_check_semaphore.acquire(blocking=False):
            return

        try:
            # Get job data for inspection
            job_data = self.send_request("get_jobs", None)

            if isinstance(job_data, dict):
                self.inspect_model(job_data.get("model_name"), job_data)
        finally:
            # Always release semaphore
            self.node_check_semaphore.release()

        # Call parent implementation to maintain original functionality
        super().check_node()

    def initialize_hosted_jobs(self):
        for model_name in FREE_MODELS:
            self.inspect_model(model_name)
