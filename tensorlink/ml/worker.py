import json
import logging
import os
import pickle
import queue
import threading
import time
import io
import base64
from collections import deque

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
        self.lock = threading.Lock()
        self.trusted = trusted

        # CUDA optimization: Check device and optimize CUDA settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_cuda_environment()

        # Mixed precision setup
        self.scaler = amp.GradScaler()
        self.use_amp = self.device.type == "cuda"  # Only use AMP with CUDA

        # Initialize CUDA streams for overlapping operations
        if self.device.type == "cuda":
            self.default_stream = torch.cuda.Stream()
            self.compute_stream = torch.cuda.Stream()
            self.memory_stream = torch.cuda.Stream()

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

    def train_loop(self):
        """Optimized training loop with CUDA awareness"""
        while not self.terminate:
            for module_id in list(self.modules.keys()):
                module = self.modules.get(module_id).to(self.device)

                # Pre-fetch next tasks when possible to overlap computation
                has_backward = not module.backward_queue.empty()
                has_forward = not module.forward_queue.empty()

                # Handle backward pass with priority (prevents gradient accumulation bottlenecks)
                if has_backward:
                    self._handle_backward(module_id, module)

                # Handle forward pass
                if has_forward:
                    with self.lock:
                        key, (size, name) = module.forward_queue.get()

                        if isinstance(key, str):
                            self._handle_generate(module_id, size, name)
                        else:
                            self._handle_forward(module_id, key, size, name)

                # Small sleep to prevent CPU hogging
                if not (has_backward or has_forward):
                    time.sleep(0.02)  # Minimal sleep when idle

    def _handle_backward(self, module_id, module):
        """Optimized backward pass with mixed precision support"""
        n_batch = module.n_batch
        next_node = module.host

        # Only process if in training mode
        if self.modules[module_id].training:
            # Critical section: lock the shared resources only when necessary
            with self.lock:
                tag, loss_relay = module.backward_queue.get()

                # Get tensor from shared memory with pinned memory for faster transfer
                tensor_bytes = get_from_shared_memory(
                    loss_relay[0], loss_relay[1], encoded=True
                )
                tensor = bytes_to_tensor(tensor_bytes)

                # Move tensors to device efficiently with pinned memory
                if self.device.type == "cuda":
                    with torch.cuda.stream(self.memory_stream):
                        # Load loss and move to the device (non-blocking for overlapping)
                        loss = attach_tensor(tensor, self.device, non_blocking=True)

                        # Retrieve intermediate values from storage
                        inter_tag = tuple(tag)
                        assoc_input, assoc_output = module.intermediates.pop(inter_tag)

                        # Move to device with memory stream to overlap transfers
                        assoc_input = assoc_input.to(self.device, non_blocking=True)
                        assoc_output = assoc_output.to(self.device, non_blocking=True)

                    # Synchronize streams before computation
                    self.memory_stream.synchronize()
                else:
                    # CPU path without non-blocking transfers
                    loss = attach_tensor(tensor, self.device)
                    inter_tag = tuple(tag)
                    assoc_input, assoc_output = module.intermediates.pop(inter_tag)
                    assoc_input = assoc_input.to(self.device)
                    assoc_output = assoc_output.to(self.device)

                # Use compute stream for backward computation
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                    with torch.cuda.stream(self.compute_stream):
                        # Scale loss for mixed precision if enabled
                        if self.use_amp:
                            # Scale the loss to prevent underflow
                            scaled_loss = self.scaler.scale(loss)
                            assoc_output.backward(scaled_loss)
                            # Unscale gradients for optimizer
                            self.scaler.unscale_(self.optimizers.get(module_id, None))
                        else:
                            assoc_output.backward(loss)
                else:
                    # CPU backward pass
                    assoc_output.backward(loss)

                if self.device.type == "cuda":
                    # Ensure backward computation is complete
                    self.compute_stream.synchronize()

                # Detach gradients and prepare for next node
                if assoc_input.grad is None:
                    dvalues = detach_tensor(
                        torch.zeros_like(assoc_input, dtype=torch.float32)
                    )
                else:
                    # Clip gradients to prevent explosion (optional)
                    if self.use_amp:
                        torch.nn.utils.clip_grad_norm_(
                            module.parameters(), max_norm=1.0
                        )
                    dvalues = detach_tensor(assoc_input.grad)

                # Clean up to avoid memory leaks
                del assoc_input, assoc_output

                # Store pass in shared memory and send to next node
                dvalues_bytes = tensor_to_bytes(dvalues)
                size, name = store_in_shared_memory(dvalues_bytes, encoded=True)
                self.send_request("send_backward", (next_node, size, name, tag))

                # Strategic memory management - clear only when necessary
                if self.device.type == "cuda":
                    # Clear only inactive tensors to avoid performance impact
                    if n_batch % 10 == 0:  # Periodic memory cleanup
                        torch.cuda.empty_cache()

    def _handle_forward(self, module_id, key, size, name):
        """Optimized forward pass with mixed precision and efficient memory management"""
        module = self.modules.get(module_id)

        # Get data from shared memory
        tensor_bytes = get_from_shared_memory(size, name, encoded=True)
        args, kwargs = tensor_bytes.split(b"|")
        args = bytes_to_tensor(args)
        kwargs = bytes_to_tensor(kwargs)

        # Move tensors to device efficiently
        if self.device.type == "cuda":
            with torch.cuda.stream(self.memory_stream):
                # Enable gradient tracking and move to device with non-blocking transfers
                inp = enable_grad(attach_tensor(args, self.device))
                kwargs = enable_grad(attach_tensor(kwargs, self.device))
            self.memory_stream.synchronize()
        else:
            # CPU path
            inp = enable_grad(attach_tensor(args, self.device))
            kwargs = enable_grad(attach_tensor(kwargs, self.device))

        # Forward pass with optional mixed precision
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            with torch.cuda.stream(self.compute_stream):
                # Use automatic mixed precision for faster computation
                if self.use_amp and module.training:
                    with amp.autocast():
                        out = module(inp, **kwargs)
                else:
                    with torch.set_grad_enabled(module.training):
                        out = module(inp, **kwargs)
        else:
            # CPU forward pass
            with torch.set_grad_enabled(module.training):
                out = module(inp, **kwargs)

        if self.device.type == "cuda":
            self.compute_stream.synchronize()

        # Store intermediate results if training (using tensor caching for efficiency)
        if self.modules[module_id].training:
            # Keep intermediates on GPU to avoid CPU<->GPU transfers
            module.intermediates[key] = [
                inp,
                handle_output(out).to(self.device, non_blocking=True),
            ]

        # Detach and store output
        detached_out = detach_tensor(out)

        # Use memory stream for data serialization and transfers
        if self.device.type == "cuda":
            with torch.cuda.stream(self.memory_stream):
                output_bytes = tensor_to_bytes(detached_out)
                size, name = store_in_shared_memory(output_bytes)
            self.memory_stream.synchronize()
        else:
            output_bytes = tensor_to_bytes(detached_out)
            size, name = store_in_shared_memory(output_bytes)

        self.send_request("send_forward", (module.host, size, name, key))

        # Incremental training counter
        if module.training:
            module.n_batch += 1

        # Strategic memory management
        if self.device.type == "cuda" and module.n_batch % 20 == 0:
            torch.cuda.empty_cache()

    def _handle_generate(self, module_id, size, name):
        """Optimized text generation with CUDA acceleration"""
        module = self.modules.get(module_id)
        query_bytes = get_from_shared_memory(size, name, encoded=True)
        args, kwargs = query_bytes.split(b"::")

        # Convert args to tensor and move to device
        input_ids = bytes_to_tensor(args)

        # Use pinned memory for faster host->device transfer
        if self.device.type == "cuda":
            # Pin memory for faster transfers
            input_ids = input_ids.pin_memory()
            input_ids = input_ids.to(self.device, non_blocking=True)
        else:
            input_ids = attach_tensor(input_ids, self.device)

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
            all_kwargs['use_cache'] = True  # Enable KV caching for faster generation

        # Synchronize for accurate profiling
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        try:
            self.send_request(
                "debug_print",
                (
                    f"DistributedWorker -> Generating with input arguments: {all_kwargs}",
                    "bright_blue",
                    logging.DEBUG,
                ),
            )
            # Generate text with the model
            with torch.no_grad():
                if isinstance(input_ids, list):
                    input_ids = input_ids[-1]

                # Use CUDA stream for generation
                if self.device.type == "cuda":
                    with torch.cuda.stream(self.compute_stream):
                        output = module.generate(input_ids, **all_kwargs)
                    self.compute_stream.synchronize()
                else:
                    output = module.generate(**all_kwargs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Detach and store generated output
            detached_out = detach_tensor(output)
            output_bytes = tensor_to_bytes(detached_out)
        except Exception as e:
            # Handle any exceptions during generation
            output_bytes = json.dumps({"error": str(e)}).encode()

        size, name = store_in_shared_memory(output_bytes)

        # Send the generated output back
        self.send_request("send_forward", (module.host, size, name, "generate"))

        # Clean memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def send_request(self, request_type, args, timeout=None):
        """Send request to coordinator node with timeout handling"""
        request = {"type": request_type, "args": args}
        try:
            self.mpc_lock.acquire(timeout=timeout)
            self.node_requests.put(request)
            response = self.node_responses.get(
                timeout=timeout
            )  # Blocking call, waits for response

        except TimeoutError as e:
            self.terminate = True

        except Exception as e:
            # print(f"Error sending request: {e}")
            response = {"return": str(e)}

        finally:
            self.mpc_lock.release()

        if response:
            return response["return"]

    def store_snapshot(self, module_id, _input, _output, epoch, micro):
        """Store model snapshot with efficient CPU transfer"""
        # Ensure the snapshots directory exists
        os.makedirs("tmp/snapshots", exist_ok=True)

        # Move to CPU before serialization to avoid GPU memory pressure
        with torch.no_grad():
            # Get parameters (state_dict) and convert tensors to a serializable format
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
        try:
            with open(file_path, "w") as f:
                json.dump(snapshot, f)
            print(f"Snapshot saved successfully: {file_path}")
        except IOError as e:
            print(f"Error saving snapshot: {e}")

    def load_module(
        self, file_name, module_id, node_id, module_name, optimizer_name, training
    ):
        """Load and prepare model with CUDA optimizations"""
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
                                # device_map="auto",
                                torch_dtype=(
                                    torch.float16 if self.use_amp else torch.float32
                                ),
                            )
                        else:
                            # Use CPU offloading for large models
                            module = AutoModelForCausalLM.from_pretrained(
                                module_name,
                                # device_map="auto",
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
                        new_state_dict[expected_key] = received_state_dict[received_key]

                    # Load remapped state dict with error handling
                    module.load_state_dict(
                        new_state_dict, strict=False
                    )  # strict=False allows minor mismatches

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
                except:
                    pass

            # Enable CUDA graph capture for repeatable operations
            if (
                hasattr(torch.cuda, 'make_graphed_callables') and False
            ):  # Disabled for now as it needs more testing
                try:
                    sample_input = torch.zeros(1, 32, device=self.device)
                    module.forward = torch.cuda.make_graphed_callables(
                        module.forward, (sample_input,)
                    )
                    print("CUDA graph optimization applied")
                except:
                    pass

        # Initialize queues and states
        module.forward_queue = queue.Queue()
        module.backward_queue = queue.Queue()
        module.intermediates = {}
        module.host = node_id
        module.n_batch = 0

        self.modules[module_id] = module
        if training:
            print("Creating Optimizer")
            optimizer_cls = get_optimizer_from_name(optimizer_name)
            # Placeholder - actual initialization happens in state_update
            self.optimizers[module_id] = optimizer_cls

        self.send_request("module_loaded", module_id)

    def check_node(self):
        """Check for node updates with efficient polling"""
        update_check_interval = 50
        counter = 0

        while not self.terminate:
            # Adaptive polling frequency based on activity
            active_modules = len(self.modules) > 0
            sleep_time = 0.05 if active_modules else 1

            if counter % update_check_interval == 0:
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
                        file_name,
                        module_id,
                        node_id,
                        module_name,
                        optimizer_name,
                        training,
                    )

                # Check for job completion/deletion requests
                elif isinstance(args, str):
                    if args in self.modules:
                        if self.modules[args].training:
                            del self.optimizers[args]
                        del self.modules[args]
                        self.send_request("debug_print", (f"Module {args} removed.",))

                # Check for node termination requests
                self.check_for_termination()

            # Process training, forward, and backward queues
            if self.modules:
                for module_id in list(
                    self.modules.keys()
                ):  # Use list to avoid modification during iteration
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

                    # Handle forward queue
                    forward_task = self.send_request("check_forward", module_id)
                    if forward_task:
                        module.forward_queue.put(forward_task)

                    # Handle backward queue
                    backward_task = self.send_request("check_backward", module_id)
                    if backward_task:
                        module.backward_queue.put(backward_task)

                    state_update = self.send_request("check_state_update", module_id)
                    if state_update:
                        with self.lock:
                            if state_update[0] == "init":
                                optimizer_kwargs = state_update[1]

                                # Configure optimizer with mixed precision support
                                if (
                                    self.use_amp
                                    and 'fused' not in optimizer_name.lower()
                                ):
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
                                            self.optimizers[
                                                module_id
                                            ] = self.optimizers[module_id](
                                                module.parameters(),
                                                **optimizer_kwargs,
                                            )
                                    else:
                                        self.optimizers[module_id] = self.optimizers[
                                            module_id
                                        ](module.parameters(), **optimizer_kwargs)
                                else:
                                    self.optimizers[module_id] = self.optimizers[
                                        module_id
                                    ](module.parameters(), **optimizer_kwargs)

                                self.send_request(
                                    "debug_print",
                                    (
                                        f"DistributedWorker -> Initialized optimizer: {optimizer_name} on {self.device.type}",
                                        "bright_blue",
                                        logging.INFO,
                                    ),
                                )
                                self.send_request(
                                    "optimizer_response", (module_id, "loaded")
                                )

                            elif state_update[0] == "step":
                                closure = state_update[1]
                                # Step optimizer with mixed precision support if using CUDA
                                if self.use_amp:
                                    # Update with scaler for mixed precision
                                    self.scaler.step(
                                        self.optimizers[module_id], closure
                                    )
                                    self.scaler.update()
                                else:
                                    self.optimizers[module_id].step(closure)

                                self.send_request(
                                    "debug_print",
                                    (
                                        "DistributedWorker -> Optimizer stepped.",
                                        "bright_blue",
                                        logging.INFO,
                                    ),
                                )
                                self.send_request(
                                    "optimizer_response", (module_id, "stepped")
                                )

                            elif state_update[0] == "zero_grad":
                                # Zero gradients with optimized memory usage
                                if self.device.type == "cuda":
                                    # More efficient for CUDA
                                    for param in module.parameters():
                                        if param.grad is not None:
                                            param.grad = None
                                else:
                                    self.optimizers[module_id].zero_grad()

                                self.send_request(
                                    "debug_print",
                                    (
                                        "DistributedWorker -> Optimizer zeroed.",
                                        "bright_blue",
                                        logging.INFO,
                                    ),
                                )
                                self.send_request(
                                    "optimizer_response", (module_id, "zeroed")
                                )

            counter += 1
            time.sleep(sleep_time)

    def check_for_termination(self):
        """Check if worker should terminate"""
        shutdown_signal = self.send_request("check_shutdown", None)
        if shutdown_signal:
            self.send_request(
                "debug_print",
                "Termination signal received. Shutting down DistributedWorker process...",
            )
            self.terminate = True

    def run(self):
        """Main execution method with robust thread management"""
        # Start node check thread
        node_check_thread = threading.Thread(target=self.check_node)
        node_check_thread.daemon = (
            True  # Make thread daemon so it exits when main thread exits
        )
        node_check_thread.start()

        # Start training loop
        try:
            self.train_loop()
        except Exception as e:
            self.send_request("debug_print", f"Error in train loop: {str(e)}")
            self.terminate = True
        finally:
            # Wait for check node thread to finish
            node_check_thread.join(timeout=5.0)

            # Final cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
