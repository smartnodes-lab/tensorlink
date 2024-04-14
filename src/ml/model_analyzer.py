import torch.nn as nn
import inspect
import torch
import ast


distributed_module_ids = []


def get_gpu_memory():
    # Check how much available memory we can allocate to the node
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
        # CPU should be able to handle 1 GB (temporary fix)
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
    assert len(indices) > 0
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
