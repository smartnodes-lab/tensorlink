import json
import logging
import os
import pickle
import queue
import threading
import time
import io
from collections import deque
import traceback

import torch
import torch.amp as amp
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForObjectDetection,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
)

from tensorlink.ml.utils import (
    get_optimizer_from_name,
    bytes_to_tensor,
    tensor_to_bytes,
    detach_tensor,
    attach_tensor,
    handle_output,
    enable_grad,
)
from tensorlink.mpc.shared_memory import get_from_shared_memory, store_in_shared_memory

base_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_TYPE_MAPPING = {
    "ForSequenceClassification": AutoModelForSequenceClassification,
    "ForTokenClassification": AutoModelForTokenClassification,
    "ForQuestionAnswering": AutoModelForQuestionAnswering,
    "ForMaskedLM": AutoModelForMaskedLM,
    "ForNextSentencePrediction": AutoModelForNextSentencePrediction,
    "ForMultipleChoice": AutoModelForMultipleChoice,
    "ForPreTraining": AutoModelForPreTraining,
    "ForCausalLM": AutoModelForCausalLM,
    "ForImageClassification": AutoModelForImageClassification,
    "ForSemanticSegmentation": AutoModelForSemanticSegmentation,
    "ForObjectDetection": AutoModelForObjectDetection,
    "ForAudioClassification": AutoModelForAudioClassification,
    "ForCTC": AutoModelForCTC,
    "ForSpeechSeq2Seq": AutoModelForSpeechSeq2Seq,
    "ForVision2Seq": AutoModelForVision2Seq,
}


class DistributedWorker:
    def __init__(self, node_requests, node_responses, mpc_lock, trusted=False):
        self.node_requests = node_requests
        self.node_responses = node_responses
        self.mpc_lock = mpc_lock
        self.rolling_buffer = deque(maxlen=10)
        self.storage_path = "./tmp/snapshots"

        self.modules = {}
        self.optimizers = {}
        self.terminate = False

        # Single lock for critical operations
        self.global_lock = threading.RLock()

        self.trusted = trusted

        # CUDA optimization: Check device and optimize CUDA settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_cuda_environment()

        # Mixed precision setup
        self.scaler = amp.GradScaler()
        self.use_amp = self.device.type == "cuda"  # Only use AMP with CUDA

        # Initialize CUDA streams for overlapping operations if using CUDA
        if self.device.type == "cuda":
            self.default_stream = torch.cuda.Stream()
            self.compute_stream = torch.cuda.Stream()
            self.memory_stream = torch.cuda.Stream()

        # Operation timings to help with scheduling
        self.last_node_check = time.time()
        self.node_check_interval = 0.05  # Check node every 50ms

    def setup_cuda_environment(self):
        """Configure optimal CUDA settings for ML workloads"""
        if self.device.type == "cuda":
            # Optimize cuDNN for stable, reproducible results
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = (
                True  # Auto-optimize convolution algorithms
            )

            # Set memory allocation strategy
            torch.cuda.empty_cache()
            # Set reasonable memory fraction to avoid OOM errors
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(
                0.85
            )  # Use up to 85% of available memory

            # Log CUDA configuration
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
            logging.info(f"Total CUDA memory: {total_memory / 1e9:.2f} GB")

    def run(self):
        """Single main loop that handles all operations sequentially"""
        try:
            logging.info("Starting DistributedWorker main loop")
            while not self.terminate:
                # First check for node updates (higher priority)
                self.check_node_updates()

                # Process any pending work in modules
                self.process_modules()

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

    def check_node_updates(self):
        """Check for updates from the node"""
        current_time = time.time()

        # Only check at specified intervals to avoid overwhelming the node
        if current_time - self.last_node_check < self.node_check_interval:
            return

        self.last_node_check = current_time

        # Check for module updates, removals, or termination signals
        args = self.send_request("check_module", None, timeout=2)

        if isinstance(args, tuple):
            # New module to load
            file_name, module_id, node_id, module_name, optimizer_name, training = args
            self.load_module(
                file_name, module_id, node_id, module_name, optimizer_name, training
            )

        elif isinstance(args, str):
            # Module to remove
            self.remove_module(args)

        # Check for termination signal
        self.check_for_termination()

    def process_modules(self):
        """Process all active modules"""
        if not self.modules:
            return

        with self.global_lock:
            module_ids = list(self.modules.keys())

        for module_id in module_ids:
            # Check if module still exists
            with self.global_lock:
                if module_id not in self.modules:
                    continue
                module = self.modules[module_id]

            # Check for parameter requests
            self.handle_parameter_request(module_id, module)

            # Handle forward pass requests
            self.handle_forward_requests(module_id, module)

            # Handle training operations if in training mode
            if hasattr(module, "training"):
                # Check training status
                is_training = self.send_request("check_train", module_id)
                if isinstance(is_training, bool):
                    module.training = is_training

                if module.training:
                    # Handle backward passes
                    self.handle_backward_requests(module_id, module)

                    # Handle optimizer updates
                    self.handle_optimizer_updates(module_id, module)

    def handle_parameter_request(self, module_id, module):
        """Handle requests for model parameters"""
        params_req = self.send_request(
            "check_parameters_request", module_id, timeout=0.5
        )

        if params_req:
            logging.info("DistributedWorker -> Sending parameters")

            # Save state dict to file with safe CPU transfer
            with open(f"parameters_{module_id}", "wb") as file:
                if self.device.type == "cuda":
                    # Temporarily move to CPU for saving
                    cpu_state_dict = {
                        k: v.detach().cpu() for k, v in module.state_dict().items()
                    }
                    torch.save(cpu_state_dict, file)
                else:
                    torch.save(module.state_dict(), file)

            self.send_request("send_parameters", (module.host, module_id))

    def handle_forward_requests(self, module_id, module):
        """Handle forward pass requests"""
        forward_task = self.send_request("check_forward", module_id)
        if forward_task:
            print(forward_task)
            if isinstance(forward_task[0], str):
                self._handle_generate(module_id, forward_task[1][0], forward_task[1][1])
            else:
                self._handle_forward(
                    module_id, forward_task[0], forward_task[1], forward_task[2]
                )

    def handle_backward_requests(self, module_id, module):
        """Handle backward pass requests"""
        backward_task = self.send_request("check_backward", module_id)
        if backward_task:
            self._handle_backward(module_id, module, backward_task[0], backward_task[1])

    def handle_optimizer_updates(self, module_id, module):
        """Handle optimizer state updates"""
        state_update = self.send_request("check_state_update", module_id)

        if not state_update:
            return

        if state_update[0] == "init":
            # Initialize optimizer
            optimizer_kwargs = state_update[1]

            if module_id in self.optimizers:
                optimizer_cls = self.optimizers[module_id]
                optimizer_name = (
                    optimizer_cls.__name__
                    if hasattr(optimizer_cls, '__name__')
                    else "Unknown"
                )

                # Create optimizer with appropriate settings for CUDA/CPU
                if self.use_amp and 'fused' not in optimizer_name.lower():
                    # Use fused implementation when available for better performance
                    if optimizer_name.lower() == 'adam':
                        try:
                            from torch.optim.adam import Adam

                            self.optimizers[module_id] = Adam(
                                module.parameters(),
                                **optimizer_kwargs,
                                fused=True,
                            )
                        except:
                            self.optimizers[module_id] = optimizer_cls(
                                module.parameters(),
                                **optimizer_kwargs,
                            )
                    else:
                        self.optimizers[module_id] = optimizer_cls(
                            module.parameters(),
                            **optimizer_kwargs,
                        )
                else:
                    self.optimizers[module_id] = optimizer_cls(
                        module.parameters(), **optimizer_kwargs
                    )

                self.send_request(
                    "debug_print",
                    (
                        f"DistributedWorker -> Initialized optimizer: {optimizer_name} on {self.device.type}",
                    ),
                    timeout=0.5,
                )

            self.send_request("optimizer_response", (module_id, "loaded"))

        elif state_update[0] == "step":
            # Step optimizer
            closure = state_update[1]

            if module_id in self.optimizers:
                if self.use_amp:
                    # Update with scaler for mixed precision
                    self.scaler.step(self.optimizers[module_id], closure)
                    self.scaler.update()
                else:
                    self.optimizers[module_id].step(closure)

                self.send_request("optimizer_response", (module_id, "stepped"))

        elif state_update[0] == "zero_grad":
            # Zero gradients
            if module_id in self.optimizers:
                if self.device.type == "cuda":
                    # More efficient for CUDA
                    for param in module.parameters():
                        if param.grad is not None:
                            param.grad = None
                else:
                    self.optimizers[module_id].zero_grad()

            self.send_request("optimizer_response", (module_id, "zeroed"))

    def _handle_backward(self, module_id, module, tag, loss_relay):
        """Process backward pass with mixed precision support"""
        if not module.training:
            return

        try:
            # Get tensor from shared memory
            tensor_bytes = get_from_shared_memory(
                loss_relay[0], loss_relay[1], encoded=True
            )
            tensor = bytes_to_tensor(tensor_bytes)

            # Move tensors to device efficiently
            if self.device.type == "cuda":
                with torch.cuda.stream(self.memory_stream):
                    loss = attach_tensor(tensor, self.device, non_blocking=True)

                    # Retrieve intermediate values from storage
                    inter_tag = tuple(tag)

                    if inter_tag not in module.intermediates:
                        logging.warning(
                            f"Missing intermediate values for tag {inter_tag}"
                        )
                        return

                    assoc_input, assoc_output = module.intermediates.pop(inter_tag)

                    # Move to device with memory stream to overlap transfers
                    assoc_input = assoc_input.to(self.device, non_blocking=True)
                    assoc_output = assoc_output.to(self.device, non_blocking=True)

                self.memory_stream.synchronize()
            else:
                # CPU path without non-blocking transfers
                loss = attach_tensor(tensor, self.device)
                inter_tag = tuple(tag)

                if inter_tag not in module.intermediates:
                    logging.warning(f"Missing intermediate values for tag {inter_tag}")
                    return

                assoc_input, assoc_output = module.intermediates.pop(inter_tag)
                assoc_input = assoc_input.to(self.device)
                assoc_output = assoc_output.to(self.device)

            # Perform backward pass
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                with torch.cuda.stream(self.compute_stream):
                    if self.use_amp:
                        # Scale loss for mixed precision
                        scaled_loss = self.scaler.scale(loss)
                        assoc_output.backward(scaled_loss)

                        if module_id in self.optimizers:
                            # Unscale gradients for optimizer
                            self.scaler.unscale_(self.optimizers[module_id])
                    else:
                        assoc_output.backward(loss)
                self.compute_stream.synchronize()
            else:
                # CPU backward pass
                assoc_output.backward(loss)

            # Detach gradients and prepare for next node
            if assoc_input.grad is None:
                dvalues = detach_tensor(
                    torch.zeros_like(assoc_input, dtype=torch.float32)
                )
            else:
                # Clip gradients to prevent explosion (optional)
                if self.use_amp:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
                dvalues = detach_tensor(assoc_input.grad)

            # Clean up to avoid memory leaks
            del assoc_input, assoc_output

            # Store pass in shared memory and send to next node
            dvalues_bytes = tensor_to_bytes(dvalues)
            size, name = store_in_shared_memory(dvalues_bytes, encoded=True)
            self.send_request("send_backward", (module.host, size, name, tag))

            # Strategic memory management - clear only when necessary
            if self.device.type == "cuda" and module.n_batch % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error in backward pass: {str(e)}")
            traceback.print_exc()

    def _handle_forward(self, module_id, key, size, name):
        """Process forward pass with mixed precision"""
        try:
            module = self.modules.get(module_id)
            if not module:
                logging.warning(f"Module {module_id} not found for forward pass")
                return

            # Get data from shared memory
            tensor_bytes = get_from_shared_memory(size, name, encoded=True)
            args, kwargs = tensor_bytes.split(b"|")
            args = bytes_to_tensor(args)
            kwargs = bytes_to_tensor(kwargs)

            # Move tensors to device efficiently
            if self.device.type == "cuda":
                with torch.cuda.stream(self.memory_stream):
                    inp = enable_grad(attach_tensor(args, self.device))
                    kwargs = enable_grad(attach_tensor(kwargs, self.device))
                self.memory_stream.synchronize()
            else:
                inp = enable_grad(attach_tensor(args, self.device))
                kwargs = enable_grad(attach_tensor(kwargs, self.device))

            # Forward pass with optional mixed precision
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                with torch.cuda.stream(self.compute_stream):
                    if self.use_amp and module.training:
                        with amp.autocast():
                            out = module(inp, **kwargs)
                    else:
                        with torch.set_grad_enabled(module.training):
                            out = module(inp, **kwargs)
                self.compute_stream.synchronize()
            else:
                with torch.set_grad_enabled(module.training):
                    out = module(inp, **kwargs)

            # Store intermediate results if training
            if module.training:
                module.intermediates[key] = [
                    inp,
                    handle_output(out).to(
                        self.device,
                        non_blocking=True if self.device.type == "cuda" else False,
                    ),
                ]

            # Detach and store output
            detached_out = detach_tensor(out)

            if self.device.type == "cuda":
                with torch.cuda.stream(self.memory_stream):
                    output_bytes = tensor_to_bytes(detached_out)
                    size, name = store_in_shared_memory(output_bytes)
                self.memory_stream.synchronize()
            else:
                output_bytes = tensor_to_bytes(detached_out)
                size, name = store_in_shared_memory(output_bytes)

            # Send result to host
            self.send_request("send_forward", (module.host, size, name, key))

            # Incremental training counter
            if module.training:
                module.n_batch += 1

            # Strategic memory management
            if self.device.type == "cuda" and module.n_batch % 20 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}")
            traceback.print_exc()

    def _handle_generate(self, module_id, size, name):
        """Process text generation with CUDA acceleration"""
        try:
            module = self.modules.get(module_id)
            if not module:
                logging.warning(f"Module {module_id} not found for generation")
                return

            query_bytes = get_from_shared_memory(size, name, encoded=True)
            args, kwargs = query_bytes.split(b"::")

            # Convert args to tensor and move to device
            input_ids = bytes_to_tensor(args)

            # Load kwargs but filter out non-generation parameters
            all_kwargs = bytes_to_tensor(kwargs)
            # Filter out other known non-generation parameters
            known_non_generation_params = ['module', 'class', 'data']
            for param in known_non_generation_params:
                all_kwargs.pop(param, None)

            # Optimize generation parameters if not specified
            if 'num_beams' not in all_kwargs and self.device.type == "cuda":
                all_kwargs['num_beams'] = (
                    4  # Use beam search for better results when on GPU
                )

            # Use efficient attention if available and not specified
            if self.device.type == "cuda" and 'use_cache' not in all_kwargs:
                all_kwargs['use_cache'] = (
                    True  # Enable KV caching for faster generation
                )

            # Generate text with the model
            with torch.no_grad():
                if isinstance(input_ids, list):
                    input_ids = input_ids[-1]

                # Use pinned memory for faster host->device transfer
                if self.device.type == "cuda":
                    input_ids = input_ids.pin_memory()
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    torch.cuda.synchronize()
                else:
                    input_ids = attach_tensor(input_ids, self.device)

                # Use CUDA stream for generation
                if self.device.type == "cuda":
                    with torch.cuda.stream(self.compute_stream):
                        output = module.generate(input_ids, **all_kwargs)
                    self.compute_stream.synchronize()
                else:
                    if not isinstance(input_ids, torch.Tensor):
                        output = module.generate(**input_ids, **all_kwargs)
                    else:
                        output = module.generate(input_ids, **all_kwargs)

            # Detach and store generated output
            detached_out = detach_tensor(output)
            output_bytes = tensor_to_bytes(detached_out)

        except Exception as e:
            # Handle any exceptions during generation
            output_bytes = json.dumps({"error": str(e)}).encode()
            logging.error(f"Error in generation: {str(e)}")
            traceback.print_exc()

        # Send the generated output back
        size, name = store_in_shared_memory(output_bytes)
        self.send_request("send_forward", (module.host, size, name, "generate"))

        # Clean memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def load_module(
        self, file_name, module_id, node_id, module_name, optimizer_name, training
    ):
        """Load and prepare model with CUDA optimizations"""
        try:
            print("Loading Model:", module_name)

            # Load the module based on trusted status
            if self.trusted:
                with open(file_name, "rb") as f:
                    module = pickle.load(f)

                    # Move model to GPU asynchronously if possible
                    if self.device.type == "cuda":
                        module = module.to(self.device, non_blocking=True)
                    else:
                        module = module.to(self.device)

            # Else try Hugging Face for model info
            elif len(module_name) > 0:
                api = HfApi()
                try:
                    # Get model information from Hugging Face api
                    api.model_info(repo_id=module_name)

                    if os.stat(file_name).st_size == 0:
                        # Optimize model loading with device map for large models
                        if self.device.type == "cuda":
                            # Check GPU memory to determine loading strategy
                            free_memory = (
                                torch.cuda.get_device_properties(0).total_memory
                                - torch.cuda.memory_allocated()
                            )

                            # If GPU has enough memory, load directly to GPU, otherwise use disk offloading
                            if free_memory > 8 * 1e9:  # 8GB threshold
                                module = AutoModelForCausalLM.from_pretrained(
                                    module_name,
                                    torch_dtype=(
                                        torch.float16 if self.use_amp else torch.float32
                                    ),
                                )
                            else:
                                # Use CPU offloading for large models
                                module = AutoModelForCausalLM.from_pretrained(
                                    module_name,
                                    offload_folder="tmp/offload",
                                    torch_dtype=(
                                        torch.float16 if self.use_amp else torch.float32
                                    ),
                                )
                        else:
                            module = AutoModelForCausalLM.from_pretrained(module_name)
                    else:
                        with open(file_name, "rb") as f:
                            metadata_size = int.from_bytes(f.read(4), "big")
                            metadata_bytes = f.read(metadata_size)
                            metadata = json.loads(metadata_bytes.decode("utf-8"))

                            state_dict_bytes = f.read()
                            state_dict_buffer = io.BytesIO(state_dict_bytes)

                            # Load with appropriate device placement
                            if self.device.type == "cuda":
                                received_state_dict = torch.load(
                                    state_dict_buffer,
                                    weights_only=True,
                                    map_location=self.device,
                                )
                            else:
                                received_state_dict = torch.load(
                                    state_dict_buffer, weights_only=True
                                )

                        # Load the expected model with optimized settings
                        if self.device.type == "cuda":
                            module = AutoModel.from_pretrained(
                                module_name,
                                device_map="auto",
                                torch_dtype=(
                                    torch.float16 if self.use_amp else torch.float32
                                ),
                            )
                        else:
                            module = AutoModel.from_pretrained(module_name)

                        model_state_dict = module.state_dict()

                        # Map received keys to expected keys
                        new_state_dict = {}
                        for expected_key, received_key in zip(
                            model_state_dict.keys(), received_state_dict.keys()
                        ):
                            new_state_dict[expected_key] = received_state_dict[
                                received_key
                            ]

                        # Load remapped state dict with error handling
                        module.load_state_dict(new_state_dict, strict=False)

                except Exception as e:
                    # TODO route error to validator for reporting
                    raise e

            else:
                # Load TorchScript model with device placement
                if self.device.type == "cuda":
                    module = torch.jit.load(file_name, map_location=self.device)
                else:
                    module = torch.jit.load(file_name)

            # Cleanup file
            os.remove(file_name)
            print("Cleaning Model...")

            # Apply model optimizations
            if self.device.type == "cuda":
                # Try converting to faster kernel implementations when possible
                if hasattr(module, 'to_bettertransformer'):
                    try:
                        module = module.to_bettertransformer()
                        print("Using BetterTransformer optimization")
                    except Exception as e:
                        logging.warning(f"Failed to apply BetterTransformer: {str(e)}")

            # Initialize state
            module.forward_queue = queue.Queue()
            module.backward_queue = queue.Queue()
            module.intermediates = {}
            module.host = node_id
            module.n_batch = 0

            # Store module and create optimizer reference
            with self.global_lock:
                self.modules[module_id] = module

                if training:
                    print("Creating Optimizer")
                    optimizer_cls = get_optimizer_from_name(optimizer_name)
                    self.optimizers[module_id] = optimizer_cls

            self.send_request("module_loaded", module_id)

        except Exception as e:
            logging.error(f"Error loading module: {str(e)}")
            traceback.print_exc()

    def remove_module(self, module_id):
        """Safely remove a module from the worker"""
        with self.global_lock:
            if module_id in self.modules:
                # Remove optimizer if it exists
                if module_id in self.optimizers:
                    del self.optimizers[module_id]

                # Remove the module
                del self.modules[module_id]

                self.send_request(
                    "debug_print",
                    (f"Module {module_id} removed.",),
                    timeout=0.5,
                )

    def check_for_termination(self):
        """Check if worker should terminate"""
        try:
            shutdown_signal = self.send_request("check_shutdown", None, timeout=1)
            if shutdown_signal:
                self.send_request(
                    "debug_print",
                    "Termination signal received. Shutting down DistributedWorker process...",
                )
                self.terminate = True

        except Exception as e:
            logging.error(f"Error checking for termination: {str(e)}")

    def send_request(self, request_type, args, timeout=None):
        """Send request to node with timeout"""
        if not hasattr(self, "hosted_models"):
            logging.debug(f"Req_type: {request_type}")

        request = {"type": request_type, "args": args}
        response = None

        try:
            # Try to acquire lock with timeout
            if not self.mpc_lock.acquire(timeout=timeout):
                logging.warning(f"Failed to acquire mpc_lock for {request_type}")
                return None

            try:
                self.node_requests.put(request)
                # Use timeout for waiting for response to avoid deadlocks
                response = self.node_responses.get(timeout=timeout)
            finally:
                self.mpc_lock.release()

        except TimeoutError as e:
            logging.error(f"Timeout in send_request for {request_type}: {str(e)}")

        except Exception as e:
            logging.error(f"Error in send_request for {request_type}: {str(e)}")
            response = {"return": str(e)}

        if response:
            return response["return"]
        return None

    def store_snapshot(self, module_id, _input, _output, epoch, micro):
        """Store model snapshot with efficient CPU transfer"""
        try:
            # Ensure the snapshots directory exists
            os.makedirs("tmp/snapshots", exist_ok=True)

            # Move to CPU before serialization to avoid GPU memory pressure
            with torch.no_grad():
                # Get module safely
                with self.global_lock:
                    if module_id not in self.modules:
                        logging.warning(f"Module {module_id} not found for snapshot")
                        return

                # Get parameters and convert tensors to a serializable format
                params = {
                    k: v.detach().cpu().numpy().tolist()
                    for k, v in self.modules[module_id].state_dict().items()
                }

                # Convert input/output tensors to CPU for serialization
                cpu_input = _input.detach().cpu().numpy().tolist()
                cpu_output = _output.detach().cpu().numpy().tolist()

            # Prepare snapshot data
            snapshot = {
                "id": module_id,
                "params": params,
                "input": cpu_input,
                "output": cpu_output,
                "epoch": epoch,
                "micro": micro,
            }

            # Define the filename
            file_path = os.path.join(
                "tmp", "snapshots", f"{module_id}_{epoch}_{micro}.json"
            )

            # Write the snapshot to a JSON file
            with open(file_path, "w") as f:
                json.dump(snapshot, f)
            logging.info(f"Snapshot saved successfully: {file_path}")

        except Exception as e:
            logging.error(f"Error saving snapshot: {str(e)}")
            traceback.print_exc()

    def _cleanup(self):
        """Clean up resources before shutdown"""
        # Clear all modules and optimizers
        with self.global_lock:
            self.modules.clear()
            self.optimizers.clear()
