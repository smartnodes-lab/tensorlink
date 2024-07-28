from transformers.utils import ModelOutput
from collections import defaultdict
import torch.nn as nn
import inspect
import torch
import ast


distributed_module_ids = []


def get_gpu_memory():
    # Check how much available mpc we can allocate to the node
    memory = 0

    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
        memory += torch.cuda.memory

        for device in devices:
            torch.cuda.set_device(device)
            memory_stats = torch.cuda.memory_stats(device)
            device_memory = memory_stats["allocated_bytes.all.peak"] / 1024 / 1024
            memory += device_memory
    else:
        # TODO CPU should be able to handle 1 GB (temporary fix)
        memory += 1.37e9

    return memory


def parameter_memory(module):
    return sum(param.numel() * param.element_size() for param in module.parameters())


def activation_memory(output: torch.Tensor):
    return output.element_size() * output.numel() if output.requires_grad else 0


def gradient_memory(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    return sum(g.numel() * g.element_size() for g in gradients) if gradients else 0


def optimizer_memory(optimizer):
    return sum(
        state.numel() * state.element_size()
        for group in optimizer.param_groups
        for state in group["params"]
    )


def estimate_memory(module):
    """
    Dummy estimate compared to estimate_memory_requirements but doesn't require a dummy
    forward pass and thus is preferred for now.
    """
    return 4 * sum(
        param.numel() * param.element_size() for param in module.parameters()
    )


def estimate_memory_requirement(layer, dummy_input: torch.Tensor, optimizer):
    layer.eval()
    output = handle_output(layer(dummy_input.detach()))
    loss = output.sum()

    optimizer = optimizer(layer.parameters())
    optimizer.zero_grad()
    loss.backward()

    params_mem = parameter_memory(layer)
    activations_mem = activation_memory(output)
    gradient_mem = gradient_memory(layer)
    optimizer_mem = optimizer_memory(optimizer)

    return output, sum([params_mem, activations_mem, gradient_mem, optimizer_mem])


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
    """Access a module from a model based on its integer ID (depth)"""
    if len(indices) <= 0:
        # First module selected
        return module, "root"

    current_module = module
    module_name = None
    for index in indices:
        children = list(current_module.named_children())
        if index >= len(children):
            raise IndexError("Index out of range for current module's children.")

        # Access the submodule
        current_module = children[index][1]
        module_name = children[index][0]
    return current_module, module_name


def detach_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach()
    elif isinstance(tensor, ModelOutput):
        for key, value in tensor.items():
            if isinstance(value, torch.Tensor):
                tensor[key] = tensor[key].detach()

        return tensor
    else:
        raise TypeError("Unsupported input type")


def enable_grad(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach()
    elif isinstance(tensor, ModelOutput):
        for key, value in tensor.items():
            if isinstance(value, torch.Tensor):
                tensor[key] = tensor[key].requires_grad_()

        return tensor
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
