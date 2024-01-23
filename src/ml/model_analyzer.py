from torchviz import make_dot
import torch.nn as nn
import inspect
import torch
import ast


class ModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        forward_args = {k: v for k, v in kwargs.items() if k in inspect.signature(self.module.forward).parameters}
        return self.module(*args, **forward_args)


def parameter_memory(module):
    return sum(param.numel() * param.element_size() for param in module.parameters())


def activation_memory(output: torch.Tensor):
    return output.element_size() * output.numel() if output.requires_grad else 0


def gradient_memory(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    return sum(g.numel() * g.element_size() for g in gradients) if gradients else 0


def optimizer_memory(optimizer):
    return sum(state.numel() * state.element_size() for group in optimizer.param_groups for state in group['params'])


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


# Distribute model to available nodes
def distribute_model(model, dummy_input, available_nodes=None):
    if available_nodes is None:
        # Test set of nodes
        available_nodes = [
            {
                "memory": 4e9,
                "latency_matrix": []
            },
            {
                "memory": 4e9,
                "latency_matrix": []
            },
            {
                "memory": 4e9,
                "latency_matrix": []
            },
            {
                "memory": 4e9,
                "latency_matrix": []
            }
        ]

    # Estimate memory requirements for the model
    _, model_mem = estimate_memory_requirement(model, dummy_input, torch.optim.Adam)
    print(f"Parent Module: {model.__class__}\n{round(model_mem / 1e9, 3)} GB")
    offloaded_memory = 0
    candidate_node = max(enumerate([node["memory"] for node in available_nodes]), key=lambda x: x[1])[0]

    if len(list(model.children())) > 0:
        for name, submodule in model.named_children():
            # dummy_input, acc = distribute_model(submodule, dummy_input)
            dummy_input, submodule_mem = estimate_memory_requirement(submodule, dummy_input, torch.optim.Adam)
            print(f"    {name}: {round(submodule_mem / 1e9, 3)} GB")

            # TODO:
            #  Priority: check for lowest latency x high memory node to offload first submodule to.
            #  Later: if submodule is too big we can call the distribute model again.
            if submodule_mem < available_nodes[candidate_node]["memory"]:
                edit_module_code(submodule)
                offloaded_memory += submodule_mem
            else:
                offloaded_memory = 0
                available_nodes = available_nodes[:candidate_node] + available_nodes[candidate_node + 1:]
                candidate_node = max(enumerate([node["memory"] for node in available_nodes]), key=lambda x: x[1])[0]


# Wrap assigned module in our distributed package
def edit_module_code(module):
    source_code = inspect.getsource(type(module))
    parsed_code = ast.parse(source_code)

    children = dict(module.named_children())

    # Search for children in the init file
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for sub_node in node.body:
                if isinstance(sub_node, ast.Assign):
                    for target in sub_node.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.attr in children.keys()
                        ):
                            # Grab original module and wrap it
                            original_module = getattr(module, target.attr)
                            wrapped_module = ModuleWrapper(original_module)
                            setattr(module, target.attr, wrapped_module)
    return module


def parse_node(node):
    id = node.split()[0]
    name = node.split("label=")[-1]
    if name[0] == "\"":
        name = name.split("\"", maxsplit=2)[1].strip()
    if name[-1] == "]":
        name = name[:-1]
    return id, name


def parse_edge(edge):
    arrow_idx = edge.find("->")
    if arrow_idx != -1:
        left_num = edge[:arrow_idx].strip()
        right_num = edge[arrow_idx + 2:].strip().split()[0]
        return left_num, right_num


# Create a graph of the module, to be used to determine connections and hidden computations in the distribution process
def create_graph(module: nn.Module, dummy_input: torch.Tensor):
    out = handle_output(module(dummy_input))

    dot = make_dot(out, params=dict(list(module.named_parameters())))

    dot_string = dot.source
    lines = dot_string.replace("\n", "").split("\t")
    nodes = [line.strip() for line in lines if "label=" in line]
    nodes = [parse_node(node) for node in nodes]
    nodes = {node[0]: node[1] for node in nodes}
    edges = [line.strip() for line in lines if "->" in line]
    edges = [parse_edge(edge) for edge in edges]

    return nodes, edges
