from transformers.utils import ModelOutput
from collections import defaultdict
import torch.nn as nn
import inspect
import torch
import ast


distributed_module_ids = []


def get_gpu_memory():
    # Check how much available mpc we can allocate to the nodes
    memory = 0

    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))

        for device in devices:
            torch.cuda.set_device(device)
            free, total = torch.cuda.memory.mem_get_info(device)
            memory += free

    else:
        # TODO CPU should be able to handle 1 GB (temporary fix)
        # memory += 4e9
        memory += 4e9

    return memory


def estimate_memory(module: nn.Module):
    """Estimate the memory usage of a module."""
    memory_usage = 0
    for param in module.parameters():
        memory_usage += param.numel() * param.element_size()

    return memory_usage


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
            "memory": 0
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
            layer_info["flops"] = 4 * seq_length * embed_dim * embed_dim + seq_length * num_heads * embed_dim
        elif isinstance(module, nn.Transformer):
            # Estimating FLOPs for Transformer model
            num_layers = module.encoder.num_layers
            embed_dim = module.d_model
            seq_length = input_shape[0]
            # Each encoder layer typically involves two self-attention layers and one feed-forward layer
            layer_info["flops"] = (4 * seq_length * embed_dim * embed_dim + seq_length * embed_dim) * num_layers

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


# Handle different wrapped outputs from huggingface models
def handle_output(tensor):
    if hasattr(tensor, "last_hidden_state"):
        tensor = tensor.last_hidden_state
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    return tensor


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
        children = list(current_module.named_children())  # Get all child modules with their names
        if index >= len(children):
            raise IndexError("Index out of range for current module's children.")

        # Access the child module at the specified index
        module_name = type(children[index][1]).__name__  # Update to the class name of the child module
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
        detached_list = type(tensor)(detach_tensor(t) if isinstance(t, (ModelOutput, torch.Tensor)) else t for t in tensor)
        del tensor
        torch.cuda.empty_cache()
        return detached_list
    elif isinstance(tensor, dict):
        detached_dict = {key: detach_tensor(value) if isinstance(value, (ModelOutput, torch.Tensor)) else value for key, value in tensor.items()}
        del tensor
        torch.cuda.empty_cache()
        return detached_dict
    else:
        raise TypeError("Unsupported input type")


def attach_tensor(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, ModelOutput):
        for key, value in tensor.items():
            if isinstance(value, torch.Tensor):
                tensor[key] = tensor[key].to(device)

        return tensor
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(attach_tensor(t, device) if isinstance(t, torch.Tensor) else t for t in tensor)
    elif isinstance(tensor, dict):
        return {key: attach_tensor(value, device) if isinstance(value, torch.Tensor) else value for key, value in
                tensor.items()}
    else:
        raise TypeError("Unsupported input type")


def enable_grad(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.is_floating_point():
            return tensor.detach().requires_grad_()  # Enable gradient for floating-point Tensors
        else:
            return tensor
    elif isinstance(tensor, ModelOutput):
        for key, value in tensor.items():
            if isinstance(value, torch.Tensor):
                tensor[key] = value.detach().requires_grad_()
        return tensor
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(enable_grad(t) if isinstance(t, (torch.Tensor, ModelOutput)) else t for t in tensor)
    elif isinstance(tensor, dict):
        return {key: enable_grad(value) if isinstance(value, (torch.Tensor, ModelOutput)) else value for key, value in tensor.items()}
    else:
        raise TypeError("Unsupported input type")


def combine_micro_batches(micro_batches):
    """
    Combines the micro-batch outputs into a single output.
    """
    if isinstance(micro_batches[0], torch.Tensor):
        # If outputs are tensors, concatenate them along the batch dimension
        return torch.cat(micro_batches, dim=0)

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
                if key == 'loss':
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
