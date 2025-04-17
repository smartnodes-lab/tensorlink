import ast
import base64
import importlib
import inspect
import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import HfApi
from transformers.utils import ModelOutput
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding


class MemoryType(Enum):
    PARAMETERS = "parameters"
    GRADIENTS = "gradients"
    ACTIVATIONS = "activations"
    OPTIMIZER = "optimizer"
    TEMPORARY = "temporary"


@dataclass
class MemoryStats:
    total_bytes: int
    breakdown: Dict[MemoryType, int]
    per_layer: Dict[str, Dict[MemoryType, int]]
    peak_memory: int
    activation_shapes: List[Tuple[str, List[int]]]


class MemoryEstimator:
    def __init__(self):
        # Memory multipliers for different optimizer types
        self.optimizer_memory_multipliers = {
            "adam": 3,  # Adam keeps 2 momentum terms + parameters
            "adamw": 3,
            "sgd": 1,  # SGD with momentum keeps 1 momentum term
            "rmsprop": 2,
            "adagrad": 2,
            "adadelta": 3,
        }

        # Activation memory factors for different layer types
        self.activation_factors = {
            nn.Conv2d: self._conv2d_activation_factor,
            nn.Linear: self._linear_activation_factor,
            nn.MultiheadAttention: self._attention_activation_factor,
            nn.TransformerEncoderLayer: self._transformer_activation_factor,
            nn.LSTM: self._lstm_activation_factor,
            nn.GRU: self._gru_activation_factor,
            nn.BatchNorm2d: 2.0,  # Running mean and variance
            nn.LayerNorm: 2.0,
            nn.Dropout: 1.0,  # Minimal overhead for mask
            nn.ReLU: 1.0,
            nn.MaxPool2d: 1.0,
            nn.AdaptiveAvgPool2d: 1.0,
        }

    def _get_dtype_size(self, dtype: torch.dtype) -> int:
        """Get size in bytes for different dtypes."""
        if dtype == torch.float32:
            return 4
        elif dtype == torch.float16 or dtype == torch.bfloat16:
            return 2
        elif dtype == torch.float64:
            return 8
        elif dtype == torch.int8 or dtype == torch.uint8:
            return 1
        elif dtype == torch.int16:
            return 2
        elif dtype == torch.int32:
            return 4
        elif dtype == torch.int64:
            return 8
        else:
            return 4  # Default to float32 size

    def _conv2d_activation_factor(self, module: nn.Conv2d, input_shape: Tuple) -> float:
        """Calculate activation factor for Conv2d layers considering intermediate feature maps."""
        _, _, h, w = input_shape
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        padding = module.padding
        stride = module.stride

        # Calculate output dimensions
        h_out = ((h + 2 * padding[0] - kernel_size[0]) // stride[0]) + 1
        w_out = ((w + 2 * padding[1] - kernel_size[1]) // stride[1]) + 1

        # Consider im2col memory overhead
        im2col_size = (
            module.in_channels * kernel_size[0] * kernel_size[1] * h_out * w_out
        )
        output_size = out_channels * h_out * w_out

        return (im2col_size + output_size) / (h * w * module.in_channels)

    def _linear_activation_factor(self, module: nn.Linear, input_shape: Tuple) -> float:
        """Calculate activation factor for Linear layers."""
        return 2.0  # Input and output activations

    def _attention_activation_factor(
        self, module: nn.MultiheadAttention, input_shape: Tuple
    ) -> float:
        """Calculate activation factor for attention layers considering Q, K, V matrices."""
        seq_length = input_shape[0]
        return 4.0 + (
            seq_length / 100
        )  # Base factor + sequence length dependent factor

    def _transformer_activation_factor(
        self, module: nn.TransformerEncoderLayer, input_shape: Tuple
    ) -> float:
        """Calculate activation factor for transformer layers."""
        return 6.0  # Multiple attention heads + FFN activations

    def _lstm_activation_factor(self, module: nn.LSTM, input_shape: Tuple) -> float:
        """Calculate activation factor for LSTM layers considering gates."""
        return 4.0  # Input, forget, cell, output gates

    def _gru_activation_factor(self, module: nn.GRU, input_shape: Tuple) -> float:
        """Calculate activation factor for GRU layers."""
        return 3.0  # Reset, update gates and new memory

    def estimate_layer_memory(
        self,
        module: nn.Module,
        input_shape: Tuple,
        batch_size: int,
        dtype: torch.dtype,
        training: bool = True,
    ) -> Dict[MemoryType, int]:
        """Estimate memory usage for a single layer."""
        memory_breakdown = {mem_type: 0 for mem_type in MemoryType}
        dtype_size = self._get_dtype_size(dtype)

        # Parameter memory
        param_memory = sum(
            p.numel() * self._get_dtype_size(p.dtype) for p in module.parameters()
        )
        memory_breakdown[MemoryType.PARAMETERS] = param_memory

        # Gradient memory during training
        if training:
            memory_breakdown[MemoryType.GRADIENTS] = param_memory

        # Activation memory
        total_elements = np.prod(input_shape) * batch_size
        base_activation_memory = total_elements * dtype_size

        # Get activation factor based on layer type
        activation_factor = 1.0
        for layer_type, factor_func in self.activation_factors.items():
            if isinstance(module, layer_type):
                activation_factor = (
                    factor_func(module, input_shape)
                    if callable(factor_func)
                    else factor_func
                )
                break

        memory_breakdown[MemoryType.ACTIVATIONS] = int(
            base_activation_memory * activation_factor
        )

        # Temporary memory buffers (for intermediate computations)
        memory_breakdown[MemoryType.TEMPORARY] = int(
            base_activation_memory * 0.5
        )  # Conservative estimate

        return memory_breakdown

    def estimate_model_memory(
        self,
        model: nn.Module,
        input_shape: Tuple,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        optimizer_type: str = "adam",
        training: bool = True,
    ) -> MemoryStats:
        """
        Estimate total memory usage for the entire model.

        Args:
            model: The PyTorch model
            input_shape: Input tensor shape (excluding batch dimension)
            batch_size: Training batch size
            dtype: Model's data type
            optimizer_type: Type of optimizer used
            training: Whether the model is in training mode

        Returns:
            MemoryStats object containing detailed memory analysis
        """
        total_memory = 0
        memory_breakdown = {mem_type: 0 for mem_type in MemoryType}
        per_layer_memory = {}
        activation_shapes = []

        # Track cumulative activation memory for peak estimation
        cumulative_activation = 0
        peak_memory = 0

        def _analyze_module(module: nn.Module, name: str, curr_input_shape: Tuple):
            nonlocal total_memory, cumulative_activation, peak_memory

            # Skip container modules
            if len(list(module.children())) > 0:
                return curr_input_shape

            # Estimate memory for current layer
            layer_memory = self.estimate_layer_memory(
                module, curr_input_shape, batch_size, dtype, training
            )

            # Update statistics
            per_layer_memory[name] = layer_memory
            for mem_type, amount in layer_memory.items():
                memory_breakdown[mem_type] += amount
                total_memory += amount

            # Track activation memory for peak estimation
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
                cumulative_activation += layer_memory[MemoryType.ACTIVATIONS]
                peak_memory = max(peak_memory, cumulative_activation)

            # Calculate output shape
            output_shape = self._calculate_output_shape(module, curr_input_shape)
            activation_shapes.append((name, list(output_shape)))

            return output_shape

        # Recursively analyze model
        curr_shape = input_shape
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                curr_shape = _analyze_module(module, name, curr_shape)

        # Add optimizer memory if in training mode
        if training and optimizer_type in self.optimizer_memory_multipliers:
            optimizer_memory = (
                memory_breakdown[MemoryType.PARAMETERS]
                * self.optimizer_memory_multipliers[optimizer_type]
            )
            memory_breakdown[MemoryType.OPTIMIZER] = optimizer_memory
            total_memory += optimizer_memory

        return MemoryStats(
            total_bytes=total_memory,
            breakdown=memory_breakdown,
            per_layer=per_layer_memory,
            peak_memory=peak_memory,
            activation_shapes=activation_shapes,
        )

    def _calculate_output_shape(self, module: nn.Module, input_shape: Tuple) -> Tuple:
        """Calculate output shape for different layer types."""
        if isinstance(module, nn.Conv2d):
            _, c, h, w = input_shape
            padding = (
                module.padding
                if isinstance(module.padding, tuple)
                else (module.padding, module.padding)
            )
            stride = (
                module.stride
                if isinstance(module.stride, tuple)
                else (module.stride, module.stride)
            )
            kernel_size = (
                module.kernel_size
                if isinstance(module.kernel_size, tuple)
                else (module.kernel_size, module.kernel_size)
            )

            h_out = ((h + 2 * padding[0] - kernel_size[0]) // stride[0]) + 1
            w_out = ((w + 2 * padding[1] - kernel_size[1]) // stride[1]) + 1
            return (module.out_channels, h_out, w_out)

        elif isinstance(module, nn.Linear):
            return (module.out_features,)

        elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
            _, c, h, w = input_shape
            kernel_size = (
                module.kernel_size
                if isinstance(module.kernel_size, tuple)
                else (module.kernel_size, module.kernel_size)
            )
            stride = (
                module.stride
                if isinstance(module.stride, tuple)
                else (module.stride, module.stride)
            )
            h_out = ((h - kernel_size[0]) // stride[0]) + 1
            w_out = ((w - kernel_size[1]) // stride[1]) + 1
            return (c, h_out, w_out)

        return input_shape


def format_memory_size(number: int) -> str:
    """Format memory size in readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if number < 1024:
            return f"{number:.2f} {unit}"
        number /= 1024
    return f"{number:.2f} TB"


def estimate_hf_model_memory(
    model_name: str,
    batch_size: int = 32,
    dtype: torch.dtype = torch.float32,
    optimizer_type: str = "adam",
    precision: str = "fp32",
    seq_length: int = 256,
    training: bool = True,
) -> Tuple[float, float]:
    api = HfApi()

    try:
        training_multiplier = 4 if training else 1
        model_info = api.model_info(repo_id=model_name)
        total_ram = model_info.usedStorage * training_multiplier
        total_vram = total_ram

        # Extract model size (if available)
        # num_params = model_info.safetensors.get("parameters", {}).get("F16", 0) // 2
        # dtype_size = {"fp32": 4, "fp16": 2, "int8": 1}.get(precision, 4)
        # param_memory = num_params * dtype_size
        # activation_memory = batch_size * seq_length * dtype_size * 4
        # total_vram = param_memory + activation_memory
        #
        # if training:
        #     optimizer_memory = param_memory * 2
        #     total_vram += optimizer_memory
        # total_ram = 1.2 * num_params * dtype_size

        return total_vram, total_ram

    except Exception as e:
        print(f"Error fetching model info: {e}")
        return 0, 0


def get_hf_model(model_name: str, tokenizer: bool = False):
    api = HfApi()
    try:
        # Get model information from Hugging Face api
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = None

        return model, tokenizer

    except Exception as e:
        # TODO route error to validator for reporting
        return


distributed_module_ids = []


def get_gpu_memory():
    # Check how much available mpc we can allocate to the roles
    memory = 0

    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))

        for device in devices:
            torch.cuda.set_device(device)
            free, total = torch.cuda.memory.mem_get_info(device)
            memory += free

    else:
        # TODO CPU should be able to handle 1 GB? (temporary fix)
        memory += 8e9

    return memory


def estimate_memory(
    module: nn.Module,
    training: bool = True,
    batch_size: int = 256,
    max_input_size=(3, 224, 224),
):
    """
    Estimate the memory usage of a module in bytes, considering parameters and approximations
    for activations without performing a forward pass.

    Args:
        module (nn.Module): The PyTorch module.
        training (bool): True if training is required during module usage.
        batch_size (int): The batch size of input data.
        max_input_size (tuple): The size of a single input (C, H, W).

    Returns:
        int: The estimated memory usage in bytes.
    """
    element_size = next(module.parameters()).element_size()
    memory_usage = sum([p.numel() * p.element_size() for p in module.parameters()])

    # Estimate memory for gradients (same size as parameters during training)
    if training:
        memory_usage *= 2  # Gradients
        memory_usage *= 2  # Optimizer estimate

    input_size = batch_size
    for i in range(len(max_input_size)):
        input_size *= max_input_size[i]

    activation_state = input_size * element_size

    memory_usage += input_size
    memory_usage += activation_state

    return int(memory_usage)


def profile_model(model: nn.Module, input_size=(1, 3, 224, 224)):
    """
    Profile a PyTorch model to estimate overhead in terms of memory usage and FLOPs.
    Args:
        model (nn.Module): The model to be profiled.
        input_size (tuple): The input size for the model.
    Returns:
        dict: A dictionary containing the analysis of each layer.
    """
    # Dictionary to hold the analysis of each layer
    analysis = {}

    # Initialize dummy input to calculate FLOPs
    dummy_input = torch.zeros(*input_size)

    # Total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # Recursively analyze each layer
    def analyze_layer(module: nn.Module, input_shape):
        layer_info = {
            "parameters": sum(p.numel() for p in module.parameters()),
            "flops": 0,
            "memory": 0,
        }

        # Estimate memory for parameters
        for param in module.parameters():
            layer_info["memory"] += estimate_memory(param)

        # Estimate FLOPs
        if isinstance(module, nn.Linear):
            layer_info["flops"] = 2 * module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            out_channels, in_channels, kh, kw = module.weight.shape
            _, _, h, w = input_shape
            flops_per_instance = 2 * in_channels * kh * kw * out_channels
            layer_info["flops"] = flops_per_instance * h * w
        elif isinstance(module, nn.MultiheadAttention):
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            seq_length = input_shape[0]  # assuming (seq_len, batch_size, embed_dim)
            # Rough estimation for self-attention FLOPs
            layer_info["flops"] = (
                4 * seq_length * embed_dim * embed_dim
                + seq_length * num_heads * embed_dim
            )
        elif isinstance(module, nn.Transformer):
            # Estimating FLOPs for Transformer model
            num_layers = module.encoder.num_layers
            embed_dim = module.d_model
            seq_length = input_shape[0]
            # Each encoder layer typically involves two self-attention layers and one feed-forward layer
            layer_info["flops"] = (
                4 * seq_length * embed_dim * embed_dim + seq_length * embed_dim
            ) * num_layers

        # Add the layer analysis to the global analysis dictionary
        analysis[module] = layer_info

        return layer_info

    # Traverse the model and analyze each layer
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            layer_input_shape = dummy_input.shape
            layer_info = analyze_layer(module, layer_input_shape)
            print(f"{name}: {layer_info}")

    return analysis


def get_first_layer(model: nn.Module):
    if len(list(model.children())) > 0:
        submodule = next(model.children())
        return get_first_layer(submodule)
    else:
        return model


def find_module(module: nn.Module, target_name: str, ids: list = []):
    if not list(module.named_children()):
        return
    children = list(module.named_children())
    for i in range(len(children)):
        name, values = children[i]
        new_ids = ids + [i]
        if name == target_name:
            return values, new_ids
        res = find_module(values, target_name, new_ids)
        if res:
            return res


def access_module(module: nn.Module, indices: list):
    """Access a module from a model based on its integer ID (depth) and return the module class name."""
    if indices == [-1]:
        # If -1 is passed, return the root module and its class name
        return module, type(module).__name__

    current_module = module
    module_name = type(module).__name__  # Set the root module's class name

    for index in indices:
        children = list(
            current_module.named_children()
        )  # Get all child modules with their names
        if index >= len(children):
            raise IndexError("Index out of range for current module's children.")

        # Access the child module at the specified index
        module_name = type(
            children[index][1]
        ).__name__  # Update to the class name of the child module
        current_module = children[index][1]  # Get the actual child module

    return current_module, module_name


def detach_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        detached_tensor = tensor.detach().cpu()
        del tensor
        torch.cuda.empty_cache()
        return detached_tensor
    elif isinstance(tensor, ModelOutput):
        for key, value in tensor.items():
            if isinstance(value, torch.Tensor):
                tensor[key] = value.detach().cpu()
                del value
                torch.cuda.empty_cache()
        return tensor
    elif isinstance(tensor, (list, tuple)):
        detached_list = type(tensor)(
            detach_tensor(t) if isinstance(t, (ModelOutput, torch.Tensor)) else t
            for t in tensor
        )
        del tensor
        torch.cuda.empty_cache()
        return detached_list
    elif isinstance(tensor, dict):
        detached_dict = {
            key: (
                detach_tensor(value)
                if isinstance(value, (ModelOutput, torch.Tensor))
                else value
            )
            for key, value in tensor.items()
        }
        del tensor
        torch.cuda.empty_cache()
        return detached_dict
    else:
        raise TypeError("Unsupported input type")


def attach_tensor(tensor, device):
    if hasattr(tensor, "to"):
        return tensor.to(device)

    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)

    elif isinstance(tensor, ModelOutput):
        for key, value in tensor.items():
            if isinstance(value, torch.Tensor):
                tensor[key] = tensor[key].to(device)

        return tensor

    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(
            attach_tensor(t, device) if isinstance(t, torch.Tensor) else t
            for t in tensor
        )

    elif isinstance(tensor, dict):
        return {
            key: (
                attach_tensor(value, device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in tensor.items()
        }
    else:
        raise TypeError("Unsupported input type")


def enable_grad(tensor):
    """
    Enables gradient computation on floating-point Tensors within nested structures.
    """
    if isinstance(tensor, torch.Tensor):
        # Enable grad if the tensor is a floating-point type
        if tensor.is_floating_point():
            return tensor.detach().clone().requires_grad_(True)
        return tensor

    elif isinstance(tensor, ModelOutput):
        # Iterate through ModelOutput fields, enabling grad for any Tensors
        for key, value in tensor.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                tensor[key] = value.detach().clone().requires_grad_(True)
        return tensor

    elif isinstance(tensor, (list, tuple)):
        # Recursively apply to each element in lists or tuples
        return type(tensor)(enable_grad(t) for t in tensor)

    elif isinstance(tensor, dict):
        # Recursively apply to each item in dictionaries
        return {key: enable_grad(value) for key, value in tensor.items()}

    else:
        raise TypeError(f"Unsupported input type: {type(tensor)}")


# Example usage:
# Assuming `model_output` is a ModelOutput instance with various nested structures
# enabled_output = enable_grad(model_output)


def handle_output(tensor):
    """
    Handle various output types from models, convert to their raw tensor form:
    - Check for specific attributes like `logits` and `last_hidden_state`.
    - If output is a tuple, return the first element (assumed to be the main output tensor).
    - If output is a dictionary, check common keys or return the first tensor found.
    - If it's already a tensor, return as-is.
    """
    if hasattr(tensor, "logits"):
        return tensor.logits
    elif hasattr(tensor, "last_hidden_state"):
        return tensor.last_hidden_state
    elif isinstance(tensor, tuple):
        if len(tensor) > 0:
            return tensor[0] if isinstance(tensor[0], torch.Tensor) else tensor
        return tensor
    elif isinstance(tensor, dict):
        # Look for common keys like 'logits' or 'last_hidden_state'
        for key in ["logits", "last_hidden_state"]:
            if key in tensor and isinstance(tensor[key], torch.Tensor):
                return tensor[key]
        # Fallback to first tensor found in dict
        for value in tensor.values():
            if isinstance(value, torch.Tensor):
                return value
    elif isinstance(tensor, torch.Tensor):
        return tensor
    raise ValueError("Unsupported output format: could not find a tensor.")


def combine_micro_batches(micro_batches):
    """
    Combines the micro-batch outputs into a single output.
    """
    if isinstance(micro_batches[0], torch.Tensor):
        # If outputs are tensors, concatenate them along the batch dimension
        return torch.cat(micro_batches, dim=0)

    elif isinstance(micro_batches[0], dict):
        combined_output = defaultdict(list)

        for output in micro_batches:
            for key, value in output.items():
                combined_output[key].append(value)

        final_output = {}
        for key, value in combined_output.items():
            if isinstance(value[0], torch.Tensor):
                # Handle scalar tensors separately
                if value[0].dim() == 0:
                    final_output[key] = torch.stack(value)
                else:
                    final_output[key] = torch.cat(value, dim=0)
            else:
                final_output[key] = value  # Leave non-tensor values as-is

        return final_output

    elif isinstance(micro_batches[0], ModelOutput):
        combined_output = defaultdict(list)

        for output in micro_batches:
            for key, value in output.items():
                combined_output[key].append(value)

        # Concatenate fields that are tensors
        final_output = {}
        micro_loss = None

        for key, value in combined_output.items():
            if isinstance(value[0], torch.Tensor):
                # Handle zero-dimensional tensors
                if key == "loss":
                    # Average the loss and store individual losses for backward pass
                    averaged_loss = torch.mean(torch.stack(value))
                    setattr(averaged_loss, "micro_loss", value)
                    final_output[key] = averaged_loss

                elif value[0].dim() == 0:
                    final_output[key] = torch.stack(value)
                else:
                    final_output[key] = torch.cat(value, dim=0)
            else:
                final_output[key] = value  # Leave as is if not a tensor

        return type(micro_batches[0])(**final_output)

    else:
        raise TypeError("Unsupported output type")


def replace_output_with_custom_grad(combined_output, custom_grad_output):
    """
    Replace the main output tensor (logits, last_hidden_state, etc.) in the combined_output
    with the custom_grad_output, preserving the original structure.
    """
    if hasattr(combined_output, "logits"):
        return combined_output.__class__(
            **{**combined_output, "logits": custom_grad_output}
        )
    elif hasattr(combined_output, "last_hidden_state"):
        return combined_output.__class__(
            **{**combined_output, "last_hidden_state": custom_grad_output}
        )
    elif isinstance(combined_output, torch.Tensor):
        return custom_grad_output
    else:
        # For custom ModelOutput-like structures, replace the first tensor found
        for key, value in combined_output.items():
            if isinstance(value, torch.Tensor):
                combined_output[key] = custom_grad_output
                break
        return combined_output


def split_into_micro_batches(combined_output, n_micro_batch):
    """
    Splits the combined output back into individual micro-batches.
    """
    if isinstance(combined_output, torch.Tensor):
        # Split the tensor along the batch dimension
        return torch.chunk(combined_output, n_micro_batch, dim=0)

    elif isinstance(combined_output, ModelOutput):
        micro_batches = [defaultdict(list) for _ in range(n_micro_batch)]

        # Iterate over each key-value pair in the combined output
        for key, value in combined_output.items():
            if isinstance(value, torch.Tensor):
                # Split the tensor along the batch dimension
                split_values = torch.chunk(value, n_micro_batch, dim=0)

                # Assign each split to the corresponding micro-batch
                for i, split_value in enumerate(split_values):
                    micro_batches[i][key] = split_value

            else:
                # Distribute non-tensor values as they are
                for i in range(n_micro_batch):
                    micro_batches[i][key] = value

        # Convert each dictionary back into a ModelOutput
        return [type(combined_output)(**batch) for batch in micro_batches]

    else:
        raise TypeError("Unsupported output type")


def get_batch_size(inputs):
    """
    Returns the batch size from the inputs to the forward pass.
    Handles both tensor and ModelOutput types.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.size(0)
    elif isinstance(inputs, ModelOutput):
        for value in inputs.values():
            if isinstance(value, torch.Tensor):
                return value.size(0)
    else:
        raise ValueError("Unsupported input type")


def chunk(inputs, chunks):
    """
    Chunks the inputs into the specified number of chunks.
    Handles both tensor and ModelOutput types.
    """

    if isinstance(inputs, torch.Tensor):
        return torch.chunk(inputs, chunks)

    elif isinstance(inputs, ModelOutput):
        chunked_outputs = []
        for i in range(chunks):
            chunked_outputs.append({})

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value_chunks = torch.chunk(value, chunks)
                for i in range(chunks):
                    chunked_outputs[i][key] = value_chunks[i]

        return [ModelOutput(**chunk) for chunk in chunked_outputs]

    elif isinstance(inputs, dict):
        chunked_dicts = [{} for _ in range(chunks)]

        for key, value in inputs.items():
            value_chunks = chunk(value, chunks)

            if value is None:
                for i in range(chunks):
                    chunked_dicts[i][key] = None
            else:
                value_chunks = chunk(value, chunks)
                for i in range(chunks):
                    chunked_dicts[i][key] = value_chunks[i]

        return chunked_dicts

    else:
        return inputs


def get_optimizer_from_name(optimizer_name):
    optimizer_class = None

    module, name = optimizer_name.rsplit(".", 1)

    # Check in torch.optim
    if "torch.optim" in module:
        optimizer_class = getattr(__import__(module, fromlist=[optimizer_name]), name)

    # Check in transformers.optimization
    elif "transformers.optimization" in module:
        optimizer_class = getattr(__import__(module, fromlist=[optimizer_name]), name)

    if optimizer_class is None:
        raise ValueError(f"Optimizer class '{optimizer_name}' not found.")

    return optimizer_class


def get_optimizer_class(optimizer_name):
    """Find and return the optimizer class from its fully qualified name."""
    optimizer_class = None

    module, name = optimizer_name.rsplit(".", 1)

    # Check in torch.optim
    if "torch.optim" in module:
        optimizer_class = getattr(__import__(module, fromlist=[optimizer_name]), name)

    # Check in transformers.optimization
    elif "transformers.optimization" in module:
        optimizer_class = getattr(__import__(module, fromlist=[optimizer_name]), name)

    if optimizer_class is None:
        raise ValueError(f"Optimizer class '{optimizer_name}' not found.")

    return optimizer_class


def tensor_to_bytes(tensor):
    """Serialize tensor or tensor-like structures into bytes, including dtype."""

    def _serialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj

        elif isinstance(obj, torch.Tensor):
            return {
                "__serialized__": True,
                "module": obj.__class__.__module__,
                "class": obj.__class__.__name__,
                "dtype": str(obj.dtype),  # Save dtype as string
                "data": obj.tolist(),
            }

        elif hasattr(obj, "__class__") and obj.__class__.__name__ == "BatchEncoding":
            return {
                "__serialized__": True,
                "module": obj.__class__.__module__,
                "class": obj.__class__.__name__,
                "data": {k: _serialize(v) for k, v in obj.items()},
            }

        # Handle tokenizers (like Qwen2TokenizerFast)
        elif (
            hasattr(obj, "vocab_size")
            and hasattr(obj, "special_tokens")
            and hasattr(obj, "get_vocab")
        ):
            # Extract essential tokenizer metadata
            tokenizer_data = {
                "name_or_path": getattr(obj, "name_or_path", None),
                "vocab_size": getattr(obj, "vocab_size", None),
                "model_max_length": getattr(obj, "model_max_length", None),
                "is_fast": getattr(obj, "is_fast", None),
                "padding_side": getattr(obj, "padding_side", None),
                "truncation_side": getattr(obj, "truncation_side", None),
                "special_tokens": {
                    k: _serialize(v)
                    for k, v in getattr(obj, "special_tokens", {}).items()
                },
                "added_tokens": [],  # We'll store essential info about added tokens
                "class_name": obj.__class__.__name__,
                "module_name": obj.__class__.__module__,
            }

            # Capture essential added token info
            if hasattr(obj, "added_tokens_decoder") and getattr(
                obj, "added_tokens_decoder", None
            ):
                for token_id, token in obj.added_tokens_decoder.items():
                    if hasattr(token, "content"):
                        tokenizer_data["added_tokens"].append(
                            {
                                "id": token_id,
                                "content": token.content,
                                "special": getattr(token, "special", False),
                            }
                        )

            return {
                "__serialized__": True,
                "module": "tokenizers",
                "class": "Tokenizer",
                "is_tokenizer": True,
                "data": tokenizer_data,
            }

        # Handle stopping criteria objects that may contain tokenizers
        elif hasattr(obj, "__class__") and "StoppingCriteria" in obj.__class__.__name__:
            serialized_data = {
                "__serialized__": True,
                "module": obj.__class__.__module__,
                "class": obj.__class__.__name__,
                "is_stopping_criteria": True,
                "data": {},
            }

            # Special handling for StringStoppingCriteria which requires tokenizer
            if obj.__class__.__name__ == "StringStoppingCriteria" and hasattr(
                obj, "tokenizer"
            ):
                serialized_data["data"]["tokenizer"] = _serialize(obj.tokenizer)

            # Serialize all attributes that can be serialized
            for attr_name in dir(obj):
                # Skip private attributes and methods
                if attr_name.startswith("_") or callable(getattr(obj, attr_name)):
                    continue

                try:
                    attr_value = getattr(obj, attr_name)
                    serialized_data["data"][attr_name] = _serialize(attr_value)
                except (ValueError, TypeError):
                    # If we can't serialize an attribute, log it but continue
                    serialized_data["data"][
                        attr_name
                    ] = f"UNSERIALIZABLE_OBJECT_{type(attr_value).__name__}"

            return serialized_data

        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}

        elif isinstance(obj, (list, tuple)):
            obj_type = list if isinstance(obj, list) else tuple
            return obj_type([_serialize(v) for v in obj])

        # Handle other objects by trying to extract their __dict__
        elif hasattr(obj, "__dict__"):
            try:
                return {
                    "__serialized__": True,
                    "module": obj.__class__.__module__,
                    "class": obj.__class__.__name__,
                    "is_object": True,
                    "data": _serialize(obj.__dict__),
                }
            except (ValueError, TypeError):
                pass

        # Return a placeholder for unsupported types
        return f"UNSERIALIZABLE_OBJECT_{type(obj).__name__}"

    return json.dumps(_serialize(tensor)).encode("utf-8")


def bytes_to_tensor(tensor_data):
    """Deserialize bytes or JSON-like dicts/lists into tensors, tokenizers, and other objects."""
    if isinstance(tensor_data, bytes):
        tensor_data = json.loads(tensor_data)

    def _deserialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj

        # Handle strings that represent unserializable objects
        if isinstance(obj, str) and obj.startswith("UNSERIALIZABLE_OBJECT_"):
            # Return None or a placeholder for unserializable objects
            return None

        if isinstance(obj, list):
            return [_deserialize(v) for v in obj]

        if isinstance(obj, dict):
            # Special handling for tokenizer metadata
            if obj.get("__tokenizer_metadata__"):
                try:
                    # Import the tokenizer class from transformers
                    from transformers import AutoTokenizer

                    # Get the name or path to load from
                    name_or_path = obj.get("name_or_path")
                    if not name_or_path:
                        # If no name_or_path, return the metadata as is
                        return obj

                    # Safely load the tokenizer using AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
                    return tokenizer
                except (ImportError, Exception) as e:
                    # Fall back to just returning the metadata
                    return obj

            # Handle serialized objects
            elif obj.get("__serialized__"):
                module_name = obj.get("module")
                cls_name = obj.get("class")
                data = obj.get("data")

                # Handle tensors
                if cls_name == "Tensor" or cls_name.endswith("Tensor"):
                    dtype_str = obj["dtype"]
                    if "." in dtype_str:
                        dtype_str = dtype_str.split(".")[
                            -1
                        ]  # Remove 'torch.' if present
                    return torch.tensor(data, dtype=getattr(torch, dtype_str))

                # Handle BatchEncoding objects
                elif cls_name == "BatchEncoding":
                    from transformers import BatchEncoding

                    return BatchEncoding({k: _deserialize(v) for k, v in data.items()})

                # Handle stopping criteria objects
                elif obj.get("is_stopping_criteria"):
                    try:
                        # First, deserialize all the data to handle nested objects like tokenizers
                        deserialized_data = {
                            k: _deserialize(v) for k, v in data.items()
                        }

                        # Try to import the stopping criteria class
                        module_parts = module_name.split(".")
                        criteria_module = __import__(module_parts[0])
                        for part in module_parts[1:]:
                            try:
                                criteria_module = getattr(criteria_module, part)
                            except AttributeError:
                                # If module path is invalid, just return the deserialized data
                                return deserialized_data

                        try:
                            criteria_class = getattr(criteria_module, cls_name)
                        except AttributeError:
                            # If class doesn't exist, return the data
                            return deserialized_data

                        # Special handling for StringStoppingCriteria which needs tokenizer
                        if cls_name == "StringStoppingCriteria":
                            # Extract tokenizer and stop_string
                            tokenizer = deserialized_data.get("tokenizer")
                            stop_string = deserialized_data.get("stop_string", "")

                            if tokenizer is None:
                                # Fall back to dictionary since we can't instantiate without tokenizer
                                return deserialized_data

                            try:
                                # Create the criteria with tokenizer and stop_string
                                criteria_instance = criteria_class(
                                    tokenizer, stop_string
                                )

                                # Set any additional attributes
                                for attr_name, attr_value in deserialized_data.items():
                                    if attr_name not in ["tokenizer", "stop_string"]:
                                        try:
                                            setattr(
                                                criteria_instance, attr_name, attr_value
                                            )
                                        except (AttributeError, TypeError):
                                            pass

                                return criteria_instance
                            except Exception:
                                # If instantiation fails for any reason, return the data
                                return deserialized_data
                        else:
                            # For other stopping criteria, try constructor with deserialized data
                            try:
                                import inspect

                                sig = inspect.signature(criteria_class.__init__)
                                param_names = list(sig.parameters.keys())[
                                    1:
                                ]  # Skip 'self'

                                # Prepare constructor arguments
                                kwargs = {}
                                for param in param_names:
                                    if param in deserialized_data:
                                        kwargs[param] = deserialized_data[param]

                                # Create the instance
                                criteria_instance = criteria_class(**kwargs)

                                # Set any remaining attributes that weren't in the constructor
                                for attr_name, attr_value in deserialized_data.items():
                                    if (
                                        attr_name not in param_names
                                        and not attr_name.startswith("_")
                                    ):
                                        try:
                                            setattr(
                                                criteria_instance, attr_name, attr_value
                                            )
                                        except (AttributeError, TypeError):
                                            pass

                                return criteria_instance
                            except Exception:
                                # If instantiation fails, return the data
                                return deserialized_data
                    except Exception:
                        # Fall back to a dictionary if reconstruction fails
                        return {k: _deserialize(v) for k, v in data.items()}

                # For other serialized objects, return dictionary representation
                return {
                    k: _deserialize(v) for k, v in data.items() if k != "__serialized__"
                }

            # Handle regular dictionaries
            return {k: _deserialize(v) for k, v in obj.items()}

        return obj

    return _deserialize(tensor_data)
