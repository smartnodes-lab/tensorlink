import json
import logging
import os
import pickle
import io
import time
import glob
import torch
import torch.amp as amp
from accelerate import init_empty_weights
from typing import Optional, List, Dict, Any
from transformers import AutoConfig, AutoModel
from safetensors import safe_open
from huggingface_hub import snapshot_download, HfApi

from tensorlink.ml.utils import (
    get_optimizer_from_name,
    bytes_to_tensor,
    tensor_to_bytes,
    detach_tensor,
    attach_tensor,
    handle_output,
    enable_grad,
    get_nested_module,
)
from tensorlink.mpc.shared_memory import get_from_shared_memory, store_in_shared_memory


def _find_module_path_by_class(
    model: torch.nn.Module, class_name: str
) -> Optional[str]:
    """
    Search the model for the first submodule whose class name matches class_name.
    Returns the module path as returned by named_modules (empty string for root).
    """
    if not class_name:
        return None

    for name, mod in model.named_children():
        # skip the root empty name if it is the same class as requested
        if name == "":
            # if root has the requested class, return empty string
            if mod.__class__.__name__ == class_name:
                return name
            continue

        if mod.__class__.__name__ == class_name:
            return name

    return None


def _load_from_pytorch_bins(
    model_path: str, layer_paths: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Fallback method to load weights from pytorch_model.bin files.
    Uses local model_path from snapshot_download.
    """
    import glob

    state_dict = {}

    # Find all bin files in the model directory
    bin_files = glob.glob(os.path.join(model_path, "*.bin"))

    if not bin_files:
        raise ValueError(f"No weight files found in {model_path}")

    logging.info(f"Found {len(bin_files)} pytorch bin files")

    layer_prefixes = set()
    for path in layer_paths:
        layer_prefixes.add(path + '.')

    for bin_path in bin_files:
        logging.info(f"Loading weights from {os.path.basename(bin_path)}")

        # Load the bin file
        shard_state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

        # Extract only the weights we need
        for key, tensor in shard_state_dict.items():
            for prefix in layer_prefixes:
                if key.startswith(prefix):
                    relative_key = key[len(prefix.rstrip('.') + '.') :]
                    state_dict[relative_key] = tensor
                    break

        # Clear memory
        del shard_state_dict

    logging.info(f"Loaded {len(state_dict)} weight tensors from pytorch bins")
    return state_dict


class DistributedWorker:
    def __init__(self, node, trusted=False):
        self.node = node
        self.node_requests = node.node_requests
        self.node_responses = node.node_responses
        self.mpc_lock = node.mpc_lock
        self.storage_path = "./tmp/snapshots"

        self.modules = {}
        self.optimizers = {}
        self.terminate = False
        self.trusted = trusted

        # CUDA optimization: Check device and optimize CUDA settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_cuda_environment()

        # Mixed precision setup
        self.scaler = amp.GradScaler()
        self.use_amp = self.device.type == "cuda"  # Only use AMP with CUDA

        # Initialize CUDA streams for overlapping operations - key performance feature
        if self.device.type == "cuda":
            self.default_stream = torch.cuda.Stream()
            self.compute_stream = torch.cuda.Stream()
            self.memory_stream = torch.cuda.Stream()

        self.hf_cache_dir = os.environ.get(
            'HF_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
        )

    def setup_cuda_environment(self):
        """Configure optimal CUDA settings for ML workloads"""
        if self.device.type == "cuda":
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            # Set memory allocation strategy
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(0.85)

            # Log CUDA configuration
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
            logging.info(f"Total CUDA memory: {total_memory / 1e9:.2f} GB")

    def send_request(self, request_type, args, timeout=None):
        """Send request to coordinator node with timeout handling"""
        request = {"type": request_type, "args": args}
        try:
            self.mpc_lock.acquire(timeout=timeout)
            self.node_requests.put(request)
            response = self.node_responses.get(
                timeout=timeout
            )  # Blocking call, waits for response
            if isinstance(response, dict):
                return response["return"]
            else:
                return response

        except TimeoutError:
            self.terminate = True
        except Exception as e:
            return {"return": str(e)}
        finally:
            self.mpc_lock.release()

    def _handle_backward(self, module_id, tag, loss_relay):
        """Handle backward pass with mixed precision support"""
        module = self.modules[module_id]
        n_batch = module.n_batch
        next_node = module.host

        # Only process if in training mode
        if module.training:
            # Get tensor from shared memory
            tensor_bytes = get_from_shared_memory(
                loss_relay[0], loss_relay[1], encoded=True
            )
            tensor = bytes_to_tensor(tensor_bytes)

            # Move tensors to device
            loss = attach_tensor(tensor, self.device)

            # Retrieve intermediate values from storage
            inter_tag = tuple(tag)
            assoc_input, assoc_output = module.intermediates.pop(inter_tag)
            assoc_input = assoc_input.to(self.device)
            assoc_output = assoc_output.to(self.device)

            # Backward pass
            if self.use_amp:
                # Scale loss for mixed precision
                scaled_loss = self.scaler.scale(loss)
                assoc_output.backward(scaled_loss)
                # Unscale gradients for optimizer
                self.scaler.unscale_(self.optimizers.get(module_id, None))
            else:
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
            self.send_request("send_backward", (next_node, size, name, tag))

            # Strategic memory management - clear only when necessary
            if self.device.type == "cuda" and n_batch % 10 == 0:
                torch.cuda.empty_cache()

    def _handle_forward(self, module_id, key, size, name):
        """Handle forward pass with mixed precision"""
        module = self.modules[module_id]

        # Get data from shared memory
        tensor_bytes = get_from_shared_memory(size, name, encoded=True)
        args, kwargs = tensor_bytes.split(b"|")
        args = bytes_to_tensor(args)
        kwargs = bytes_to_tensor(kwargs)

        # Move tensors to device
        inp = enable_grad(attach_tensor(args, self.device))
        kwargs = enable_grad(attach_tensor(kwargs, self.device))

        # Forward pass
        if self.use_amp and module.training:
            with amp.autocast():
                out = module(inp, **kwargs)
        else:
            with torch.set_grad_enabled(module.training):
                out = module(inp, **kwargs)

        # Store intermediate results if training
        if module.training:
            module.intermediates[key] = [inp, handle_output(out).to(self.device)]

        # Detach and store output
        detached_out = detach_tensor(out)
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

        if isinstance(input_ids, list):
            input_ids = input_ids[-1]

        if self.device.type == "cuda":
            # Pin memory for faster transfers
            input_ids = input_ids.pin_memory()

        # Load kwargs but filter out non-generation parameters
        all_kwargs = bytes_to_tensor(kwargs)
        if hasattr(input_ids, "input_ids"):
            all_kwargs['input_ids'] = input_ids.input_ids
            if hasattr(input_ids, "attention_mask"):
                all_kwargs["attention_mask"] = input_ids.attention_mask
        else:
            all_kwargs['input_ids'] = input_ids

        all_kwargs = attach_tensor(all_kwargs, self.device)

        # Filter out other known non-generation parameters
        known_non_generation_params = ['module', 'class', 'data']
        for param in known_non_generation_params:
            all_kwargs.pop(param, None)

        # Optimize generation parameters if not specified
        if 'num_beams' not in all_kwargs and self.device.type == "cuda":
            all_kwargs['num_beams'] = 2

        # Use efficient attention if available and not specified
        if self.device.type == "cuda" and 'use_cache' not in all_kwargs:
            all_kwargs['use_cache'] = True  # Enable KV caching for faster generation

        try:
            with torch.no_grad():
                # Use pinned memory for faster host->device transfer and synchronize for accurate profiling
                if self.device.type == "cuda":
                    with torch.cuda.stream(self.compute_stream):
                        output = module.generate(**all_kwargs)
                    self.compute_stream.synchronize()

                else:
                    output = module.generate(**all_kwargs)

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

    def load_module(self, module_info: dict):
        """
        Load and prepare model from file or directly from HuggingFace.

        For direct HuggingFace loading without a file, just provide module_name.
        Default parameters allow for simplified calling when loading generic models.
        """
        module_id = module_info.get("module_id")
        model_name = module_info.get("name")
        module_name = module_info.get("module_name")
        training = module_info.get("training", False)
        our_id = module_info.get("assigned_workers")[0]
        file_name = module_id + our_id

        if module_id is None:
            raise ValueError("For standard loading, module_id must be provided")

        # Try to load the module based on trusted status
        if self.trusted:
            with open(file_name, "rb") as f:
                module = pickle.load(f)
                module = module.to(self.device)

        # Else try Hugging Face for model info
        else:
            model_config = self._load_model_skeleton(model_name, module_id)
            module = self._initialize_module_from_config(
                model_config, model_name, module_name, module_info
            )

        #   else:
        #     # Load TorchScript model
        #     if self.device.type == "cuda":
        #         module = torch.jit.load(file_name, map_location=self.device)
        #     else:
        #         module = torch.jit.load(file_name)

        # Cleanup file
        try:
            os.remove(file_name)
        except:
            pass

        # Apply model optimizations
        if self.device.type == "cuda":
            # Try converting to faster kernel implementations when possible
            if hasattr(module, 'to_bettertransformer'):
                try:
                    module = module.to_bettertransformer()
                except:
                    pass

        # Initialize storage structures
        module.intermediates = {}
        module.host = module_info.get('host')
        module.n_batch = 0

        self.modules[module_id] = module
        if training:
            optimizer_name = module_info.get("optimizer_type")
            optimizer_cls = get_optimizer_from_name(optimizer_name)
            # Placeholder - actual initialization happens in state_update
            self.optimizers[module_id] = optimizer_cls

        self.send_request("module_loaded", module_id)

    def _load_specific_layer_weights(
        self,
        model_name: str,
        layer_paths: List[str],
        single: bool = False,
        base_model_prefix: str = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load only the weights for specific layers from HuggingFace.
        Uses safetensors for efficient weight loading without loading entire model.

        If layer_paths contains 'model' or is empty, loads all weights.
        """
        state_dict = {}

        try:
            # Use snapshot_download for efficient caching
            logging.info(f"Checking cache for {model_name}")
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.hf_cache_dir,
                allow_patterns=[
                    "*.safetensors",
                    "*.bin",
                    "*.json",
                ],
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                local_files_only=False,
            )
            logging.info(f"Model located at: {model_path}")

            # Find all safetensors files
            safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

            if safetensor_files:
                logging.info(f"Found {len(safetensor_files)} safetensors files")

                # Load only specific layers
                layer_path_to_idx = {path: idx for idx, path in enumerate(layer_paths)}

                for shard_path in safetensor_files:
                    logging.info(f"Reading weights from {os.path.basename(shard_path)}")

                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        keys_loaded = 0
                        for key in f.keys():
                            # Check if key starts with any of our layer paths
                            for layer_path, layer_idx in layer_path_to_idx.items():
                                layer_prefix = layer_path + '.'
                                if key.startswith(layer_prefix):
                                    # Extract the part after the layer path
                                    new_key = key[len(layer_prefix) :]
                                    if single and '.' in new_key:
                                        # For single modules, remove one more level
                                        new_key = new_key.split('.', 1)[1]
                                    state_dict[new_key] = f.get_tensor(key)
                                    keys_loaded += 1
                                    break

                        if keys_loaded > 0:
                            logging.info(
                                f"  Loaded {keys_loaded} tensors from this shard"
                            )

                logging.info(
                    f"Total: Loaded {len(state_dict)} weight tensors for {len(layer_paths)} layers"
                )

            else:
                # Fallback: use pytorch_model.bin files
                logging.info("No safetensors found, trying pytorch_model.bin")
                bin_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))

                if bin_files:
                    for bin_path in bin_files:
                        logging.info(f"Loading from {os.path.basename(bin_path)}")
                        shard_dict = torch.load(bin_path, map_location="cpu")

                        # Load only matching keys
                        for key, value in shard_dict.items():
                            for layer_path in layer_paths:
                                layer_prefix = layer_path + '.'
                                if key.startswith(layer_prefix):
                                    new_key = key[len(layer_prefix) :]
                                    if single and '.' in new_key:
                                        new_key = new_key.split('.', 1)[1]
                                    state_dict[new_key] = value
                                    break

                    logging.info(f"Loaded {len(state_dict)} tensors from .bin files")
                else:
                    raise ValueError(f"No weight files found in {model_path}")

        except Exception as e:
            logging.error(f"Error loading weights: {e}")
            raise ValueError(f"Failed to load layer weights: {str(e)}")

        return state_dict

    def _initialize_module_from_config(
        self,
        config: AutoConfig,
        model_name: str,
        module_name: str,
        module_info: Dict[str, Any],
    ) -> torch.nn.Module:
        """
        Load model or specific layers from HuggingFace.
        Handles both single modules and grouped layer ranges.
        """
        try:
            # Determine if this is a grouped layer load
            module_type = module_info.get('type', 'offloaded')
            module_id = module_info.get("module_id")

            if module_type == 'offloaded_group':
                # Load grouped layers
                return self._load_grouped_layers(
                    model_name, config, module_id, module_info
                )
            else:
                # Load single module
                return self._load_single_module(model_name, config, module_info)

        except Exception as e:
            logging.error(f"Failed to load model from HuggingFace: {str(e)}")
            raise ValueError(f"Failed to load model from HuggingFace: {str(e)}")

    def _load_grouped_layers(
        self,
        model_name: str,
        config: AutoConfig,
        module_id: str,
        module_info: Dict[str, Any],
    ) -> torch.nn.Module:
        """
        Load a group of layers as a single module. Uses empty weights initialization
        and only loads required layer weights.
        """
        layer_paths = module_info.get('layer_paths', [])
        layer_range = module_info.get('layer_range', [])
        expected_inputs = module_info.get('expected_inputs', [])
        expected_outputs = module_info.get('expected_outputs', [])

        if not layer_paths:
            raise ValueError("layer_paths must be provided for grouped layer loading")

        logging.info(
            f"Loading grouped layers {layer_range[0]}-{layer_range[1]} from {model_name}"
        )

        # Initialize model with empty weights (no memory overhead)
        with init_empty_weights():
            base_model = AutoModel.from_config(config)

        # Create the layer group wrapper with empty weights
        grouped_module = self._create_layer_group_wrapper(
            base_model, layer_paths, expected_inputs, expected_outputs
        )

        # Get name of model for loading weights
        base_model_prefix = getattr(base_model, "base_model_prefix", None)

        # Now load only the weights for the assigned layers
        logging.info(f"Loading weights for layers {layer_range[0]}-{layer_range[1]}")
        state_dict = self._load_specific_layer_weights(
            model_name, layer_paths, base_model_prefix=base_model_prefix
        )

        # Load the state dict into the grouped module
        grouped_module = grouped_module.to_empty(device=self.device)
        missing_keys, unexpected_keys = grouped_module.load_state_dict(
            state_dict, strict=False
        )

        # Clean up base model to free memory
        del base_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logging.info(f"Successfully loaded {len(layer_paths)} layers with weights")

        return grouped_module

    def _load_single_module(
        self, model_name: str, config: AutoConfig, module_info: Dict[str, Any]
    ) -> torch.nn.Module:
        """
        Load a single module (e.g., just the RMSNorm layer).
        Uses empty weights initialization and only loads required module weights.
        """
        parent_module_path = module_info.get('parent_module_path', '')
        module_class_name = module_info.get('module', '')

        logging.info(f"Loading single module {module_class_name} from {model_name}")

        if parent_module_path == "":
            logging.info("Parent module is entire model — loading full model.")
            return AutoModel.from_config(config).to(self.device)

        # Initialize model with empty weights
        with init_empty_weights():
            base_model = AutoModel.from_config(config)

        # Extract the specific module with empty weights
        if parent_module_path and parent_module_path != "model":
            # explicit path provided
            target_module = get_nested_module(base_model, parent_module_path)
            effective_layer_path = parent_module_path
        else:
            # parent_module_path is 'model' or empty -> try to find by class name
            found_path = _find_module_path_by_class(base_model, module_class_name)
            if found_path is None:
                # if not found, as a safe fallback return the root module but warn the caller
                target_module = base_model
                effective_layer_path = parent_module_path or "model"
            else:
                effective_layer_path = f"model.{found_path}"
                target_module = get_nested_module(base_model, effective_layer_path)

        # Get name of model for loading weights
        base_model_prefix = getattr(base_model, "base_model_prefix", None)

        # Load only the weights for this specific module
        logging.info(f"Loading weights for {parent_module_path}")

        state_dict = self._load_specific_layer_weights(
            model_name,
            [effective_layer_path],
            single=True,
            base_model_prefix=base_model_prefix,
        )

        target_module = target_module.to_empty(device=self.device)

        # Load the state dict
        missing_keys, unexpected_keys = target_module.load_state_dict(
            state_dict, strict=False
        )

        # Move to device
        target_module = target_module.to(self.device)

        # Clean up
        del base_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logging.info(f"Successfully loaded single module {module_class_name}")

        return target_module

    def _create_layer_group_wrapper(
        self,
        base_model: torch.nn.Module,
        layer_paths: List[str],
        expected_inputs: List[str],
        expected_outputs: List[str],
    ) -> torch.nn.Module:
        """
        Create a wrapper module that processes multiple layers sequentially.
        This allows the worker to process all layers in one forward pass.
        """

        class LayerGroupModule(torch.nn.Module):
            def __init__(
                self,
                layers: List[torch.nn.Module],
                input_vars: List[str],
                output_vars: List[str],
                forward_logic: Optional[str] = None,
            ):
                super().__init__()
                self.layers = torch.nn.ModuleList(layers)
                self.input_vars = input_vars
                self.output_vars = output_vars
                self.forward_logic = forward_logic
                self.num_layers = len(layers)

            def forward(self, *args, **kwargs):
                """
                Process input through all layers sequentially.

                The parent's injected forward will call this as:
                    output1, output2, ... = worker(input1, input2, ...)

                We need to:
                1. Map positional args to the expected variable names
                2. Process through all layers
                3. Return outputs in the expected order
                """
                # Map positional args to named variables
                input_dict = {}
                for i, var_name in enumerate(self.input_vars):
                    if i < len(args):
                        input_dict[var_name] = args[i]

                # Add any keyword args
                input_dict.update(kwargs)

                # Extract the main state variable (usually hidden_states)
                # This is what flows through the layers
                hidden_states = input_dict.get('hidden_states')
                if hidden_states is None:
                    raise ValueError("Expected 'hidden_states' in inputs")

                # Initialize accumulators for outputs
                all_hidden_states = input_dict.get('all_hidden_states', ())
                all_self_attns = input_dict.get('all_self_attns', ())

                # Extract control flags
                output_hidden_states = input_dict.get('output_hidden_states', False)
                output_attentions = input_dict.get('output_attentions', False)
                use_cache = input_dict.get('use_cache', False)

                # Process through layers
                for idx, layer in enumerate(self.layers):
                    # Add current hidden state if needed
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)

                    # Prepare layer inputs from input_dict
                    layer_kwargs = {}

                    # Common layer arguments
                    for key in [
                        'attention_mask',
                        'position_ids',
                        'position_embeddings',
                        'past_key_value',
                        'cache_position',
                        'causal_mask_mapping',
                    ]:
                        if key in input_dict:
                            layer_kwargs[key] = input_dict[key]

                    # Control flags
                    if use_cache:
                        layer_kwargs['use_cache'] = use_cache
                    if output_attentions:
                        layer_kwargs['output_attentions'] = output_attentions

                    # Call the layer
                    layer_outputs = layer(hidden_states, **layer_kwargs)

                    # Handle layer outputs
                    if isinstance(layer_outputs, tuple):
                        hidden_states = layer_outputs[0]
                        if output_attentions and len(layer_outputs) > 1:
                            all_self_attns = all_self_attns + (layer_outputs[1],)
                    else:
                        hidden_states = layer_outputs

                # Build output dictionary
                output_dict = {
                    'hidden_states': hidden_states,
                    'all_hidden_states': all_hidden_states,
                    'all_self_attns': all_self_attns,
                }

                # Return outputs in the expected order
                if len(self.output_vars) == 1:
                    return output_dict[self.output_vars[0]]
                else:
                    return tuple(output_dict.get(var) for var in self.output_vars)

        # Extract the actual layer modules
        layers = [get_nested_module(base_model, path) for path in layer_paths]

        # Create and return the wrapper
        return LayerGroupModule(layers, expected_inputs, expected_outputs)

    def process_state_update(self, module_id, state_update):
        """Process optimizer state updates"""
        module = self.modules[module_id]

        if state_update[0] == "init":
            optimizer_kwargs = state_update[1]
            optimizer_name = self.optimizers[module_id].__name__

            # Configure optimizer with mixed precision support
            if self.use_amp and 'fused' not in optimizer_name.lower():
                # Use fused implementation when available for better performance
                if optimizer_name.lower() == 'adam':
                    try:
                        from torch.optim.adam import Adam

                        self.optimizers[module_id] = Adam(
                            module.parameters(), **optimizer_kwargs, fused=True
                        )
                    except:
                        self.optimizers[module_id] = self.optimizers[module_id](
                            module.parameters(), **optimizer_kwargs
                        )
                else:
                    self.optimizers[module_id] = self.optimizers[module_id](
                        module.parameters(), **optimizer_kwargs
                    )
            else:
                self.optimizers[module_id] = self.optimizers[module_id](
                    module.parameters(), **optimizer_kwargs
                )

            self.send_request(
                "debug_print",
                (
                    f"DistributedWorker -> Initialized optimizer: {optimizer_name} on {self.device.type}",
                    "bright_blue",
                    logging.INFO,
                ),
            )
            self.send_request("optimizer_response", (module_id, "loaded"))

        elif state_update[0] == "step":
            closure = state_update[1]
            # Step optimizer with mixed precision support if using CUDA
            if self.use_amp:
                # Update with scaler for mixed precision
                self.scaler.step(self.optimizers[module_id], closure)
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
            self.send_request("optimizer_response", (module_id, "stepped"))

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
                ("DistributedWorker -> Optimizer zeroed.", "bright_blue", logging.INFO),
            )
            self.send_request("optimizer_response", (module_id, "zeroed"))

    def main_loop(self):
        """Main execution loop. Sequentially executes the following tasks: check for new jobs, check for incoming data
        or model update requests, and then processes any outstanding forwards or backwards passes on the loaded modules
        """
        # Check for new modules to load

        args = self.send_request("check_module", None)

        # If we have received model info now load the model in this process
        if isinstance(args, dict):
            self.load_module(args)

        # For workers that have received model info, now load the model in this process
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

        # Process each module sequentially
        if self.modules:
            for module_id in list(self.modules.keys()):
                module = self.modules[module_id]

                # Check if module is in training mode
                is_training = self.send_request("check_train", module_id)
                if isinstance(is_training, bool):
                    module.training = is_training

                # Check for parameters requests
                params_req = self.send_request("check_parameters_request", module_id)
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
                        self._handle_generate(module_id, size, name)
                    else:
                        self._handle_forward(module_id, key, size, name)

                # Handle backward queue
                backward_task = self.send_request("check_backward", module_id)
                if backward_task:
                    tag, loss_relay = backward_task
                    self._handle_backward(module_id, tag, loss_relay)

        # Small sleep to prevent CPU hogging
        time.sleep(0.001)

    def _load_model_skeleton(self, model_name: str, module_id: str):
        """Load the full model structure with empty weights"""
        model_config = AutoConfig.from_pretrained(model_name)
        with init_empty_weights():
            skeleton_model = AutoModel.from_config(model_config)

        skeleton_model.eval()  # Set to eval mode initially
        self.modules[module_id] = skeleton_model
        return model_config

    def run(self):
        """Main execution thread"""
        while not self.terminate:
            self.main_loop()
            time.sleep(0.001)

        # Final cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
