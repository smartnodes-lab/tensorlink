from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.utils import estimate_hf_model_memory, get_hf_model
from tensorlink.ml.worker import DistributedWorker

import torch
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
        """Main execution method - simplified sequential approach"""
        while not self.terminate:
            # Check for new modules to load
            args = self.send_request("check_module", None)
            if isinstance(args, tuple):
                (
                    file_name,
                    module_id,
                    node_id,
                    module_name,
                    optimizer_name,
                    training,
                ) = args
                self.load_module(
                    file_name, module_id, node_id, module_name, optimizer_name, training
                )
            # Check for job completion/deletion requests
            elif isinstance(args, str):
                if args in self.modules:
                    if self.modules[args].training:
                        del self.optimizers[args]
                    del self.modules[args]
                    self.send_request("debug_print", (f"Module {args} removed.",))

            # Check for termination request
            shutdown_signal = self.send_request("check_shutdown", None)
            if shutdown_signal:
                self.send_request(
                    "debug_print",
                    "Termination signal received. Shutting down DistributedWorker process...",
                )
                self.terminate = True
                break

            # Process each module sequentially
            if self.modules:
                for module_id in list(self.modules.keys()):
                    module = self.modules[module_id]

                    # Check if module is in training mode
                    is_training = self.send_request("check_train", module_id)
                    if isinstance(is_training, bool):
                        module.training = is_training

                    # Check for parameters requests
                    params_req = self.send_request(
                        "check_parameters_request", module_id
                    )
                    if params_req:
                        self.send_request(
                            "debug_print", ("DistributedWorker -> Sending parameters.",)
                        )
                        # Save state dict to file
                        with open(f"parameters_{module_id}", "wb") as file:
                            # Optimize CPU transfer if needed
                            if self.device.type == "cuda":
                                # Temporarily move to CPU for saving
                                cpu_state_dict = {
                                    k: v.detach().cpu()
                                    for k, v in module.state_dict().items()
                                }
                                torch.save(cpu_state_dict, file)
                            else:
                                torch.save(module.state_dict(), file)

                        self.send_request("send_parameters", (module.host, module_id))

                    # Handle state updates
                    state_update = self.send_request("check_state_update", module_id)
                    if state_update:
                        self.process_state_update(module_id, state_update)

                    # Handle forward queue
                    forward_task = self.send_request("check_forward", module_id)
                    if forward_task:
                        key, (size, name) = forward_task
                        if isinstance(key, str):
                            self.handle_generate(module_id, size, name)
                        else:
                            self.handle_forward(module_id, key, size, name)

                    # Handle backward queue
                    backward_task = self.send_request("check_backward", module_id)
                    if backward_task:
                        tag, loss_relay = backward_task
                        self.handle_backward(module_id, tag, loss_relay)

            self.check_node()

            # Small sleep to prevent CPU hogging
            time.sleep(0.1)

        # Final cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
