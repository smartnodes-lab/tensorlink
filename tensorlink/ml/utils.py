from transformers.utils import ModelOutput
from collections import defaultdict
import torch.nn as nn
import inspect
import torch
import ast


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
        # TODO CPU should be able to handle 100 MB (temporary fix)
        # memory += 400e6
        memory += 1e9

    return memory


def estimate_memory(module: nn.Module, batch_size: int = 256, input_size=(3, 224, 224)):
    """
    Estimate the memory usage of a module in bytes, considering parameters and approximations
    for activations without performing a forward pass.

    Args:
        module (nn.Module): The PyTorch module.
        batch_size (int): The batch size of input data.
        input_size (tuple): The size of a single input (C, H, W).

    Returns:
        int: The estimated memory usage in bytes.
    """
    memory_usage = 0

    # Estimate memory for model parameters
    for param in module.parameters():
        memory_usage += param.numel() * param.element_size()

    # Estimate memory for gradients (same size as parameters during training)
    # if module.training:
    memory_usage *= 2

    # Approximate memory for input data
    input_memory = batch_size * torch.prod(torch.tensor(input_size)) * 4  # float32 inputs
    memory_usage += input_memory.item()

    # Estimate memory for activations based on layer type (rough estimation)
    activation_factor = 1.5 if isinstance(module, (nn.Conv2d, nn.Linear)) else 1.2
    activation_memory = input_memory * activation_factor * len(list(module.children()))
    memory_usage += activation_memory

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
            return tensor.clone().detach().requires_grad_()  # Enable gradient for floating-point Tensors
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
