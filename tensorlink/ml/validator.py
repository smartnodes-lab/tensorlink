from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.utils import estimate_hf_model_memory, get_hf_model
from tensorlink.ml.worker import DistributedWorker

import traceback
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
        """Inspect a model with proper coordination with other threads"""
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

            # Save the distribution with proper locking
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

    def check_node(self):
        """Check for node updates with efficient scheduling and locking and perform job inspection"""
        try:
            # Get job data for inspection
            job_data = self.send_request("get_jobs", None)

            if isinstance(job_data, dict):
                # Offload model inspection to a background thread to avoid blocking
                self.inspect_model(job_data.get("model_name"), job_data)
        except Exception as e:
            logging.error(f"Error checking for jobs: {str(e)}")

    def initialize_hosted_jobs(self):
        """Initialize hosted jobs with proper thread coordination"""
        # Use thread pool to avoid blocking the main thread
        for model_name in FREE_MODELS:
            # Schedule model inspections with small delays to avoid overloading
            # the system all at once
            self.thread_pool.submit(
                lambda name=model_name: (
                    time.sleep(0.5),  # Small delay between model initializations
                    self.inspect_model(name),
                )
            )

    def run(self):
        """Single main loop that handles all operations sequentially"""
        try:
            logging.info("Starting DistributedWorker main loop")
            while not self.terminate:
                # First check for node updates (higher priority)
                self.check_node_updates()

                # Process any pending work in modules
                self.process_modules()

                self.check_node()

                # Adaptive sleep based on activity level
                if not self.modules:
                    time.sleep(1)  # Longer sleep when idle
                else:
                    time.sleep(0.005)  # Short sleep when active

        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            traceback.print_exc()
        finally:
            self._cleanup()
            logging.info("DistributedWorker shutdown complete")
