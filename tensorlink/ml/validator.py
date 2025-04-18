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

        # Add validator-specific locks and state tracking
        self.models_lock = threading.RLock()
        self.inspection_in_progress = threading.Event()

        # Add to the thread states dictionary inherited from DistributedWorker
        self.thread_states['model_inspection'] = {
            'active': False,
            'last_run': time.time(),
        }

        # Additional semaphore for model inspection operations
        self.inspection_priority = threading.Semaphore(2)

    def inspect_model(self, model_name: str, job_data: dict = None):
        """Inspect a model with proper coordination with other threads"""
        # Set flag to indicate model inspection is happening
        self.thread_states['model_inspection']['active'] = True
        self.thread_states['model_inspection']['last_run'] = time.time()

        # Try to acquire inspection priority
        inspection_token_acquired = self.inspection_priority.acquire(blocking=False)

        try:
            # Only set the inspection flag if we're actually performing the inspection
            self.inspection_in_progress.set()

            try:
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
                    with self.models_lock:
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
            finally:
                # Always clear the inspection flag
                self.inspection_in_progress.clear()
        finally:
            # Always release inspection token if acquired
            if inspection_token_acquired:
                self.inspection_priority.release()

            # Update thread state
            self.thread_states['model_inspection']['active'] = False

    def check_node(self):
        """Check for node updates with efficient scheduling and locking and perform job inspection"""
        start_time = time.time()

        # Update tracking of node check activity (inheriting base implementation pattern)
        self.thread_states['node_check']['active'] = True
        self.thread_states['node_check']['last_run'] = start_time

        # Use the priority system from the parent class
        high_priority = self.check_counter % 5 == 0

        # Get job data first before potentially heavy base class operations
        if not self.ml_operation_in_progress.is_set() or high_priority:
            try:
                # Get job data for inspection
                job_data = self.send_request("get_jobs", None)

                if isinstance(job_data, dict):
                    # Offload model inspection to a background thread to avoid blocking
                    self.thread_pool.submit(
                        self.inspect_model, job_data.get("model_name"), job_data
                    )
            except Exception as e:
                logging.error(f"Error checking for jobs: {str(e)}")

        # Now call the parent implementation (which handles all the priority management)
        super().check_node()

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
        """Override run method to initialize hosted jobs before starting main processing"""
        try:
            # Initialize hosted jobs in a background thread
            init_thread = threading.Thread(
                target=self.initialize_hosted_jobs,
                name="InitializeHostedJobsThread",
                daemon=True,
            )
            init_thread.start()

            # Call the parent run method to use the improved implementation
            super().run()
        except Exception as e:
            logging.error(f"Error in DistributedValidator run: {str(e)}")
            traceback.print_exc()
            self.terminate = True
