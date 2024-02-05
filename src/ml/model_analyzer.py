from torchviz import make_dot
import torch.nn as nn
import inspect
import torch
import ast


def parameter_memory(module):
    return sum(param.numel() * param.element_size() for param in module.parameters())


def activation_memory(output: torch.Tensor):
    return output.element_size() * output.numel() if output.requires_grad else 0


def gradient_memory(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    return sum(g.numel() * g.element_size() for g in gradients) if gradients else 0


def optimizer_memory(optimizer):
    return sum(state.numel() * state.element_size() for group in optimizer.param_groups for state in group['params'])


def estimate_memory(module):
    """
    Dummy estimate compared to estimate_memory_requirements but doesn't require a dummy
    forward pass and thus is preferred for now.
    """
    return 4 * sum(param.numel() * param.element_size() for param in module.parameters())


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


def inspect_module_code(self, model: nn.Module, submodules):
    """
    Distribute model to available connected nodes, assign modules based on memory requirements & latency
    """

    # Placeholder for method to grab candidate nodes from the network
    # available_nodes = self.all_nodes
    # candidate_node = max(enumerate([node["memory"] for node in available_nodes]), key=lambda x: x[1])[0]
    candidate_node = self.all_nodes[0]  # Placeholder
    candidate_node_memory = 1.4e9  # keeps track of offloaded memory to node

    # Grab model source code
    source_code = inspect.getsource(type(model))
    parsed_code = ast.parse(source_code)
    children = dict(model.named_children())

    for node in ast.walk(parsed_code):
        # Identify init method of model
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for sub_node in node.body:
                if isinstance(sub_node, ast.Assign):
                    for target in sub_node.targets:
                        # Find modules in init method that match the name of the named_children
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.attr in children.keys()
                        ):
                            # Get the original module + required memory
                            original_module = getattr(model, target.attr)
                            module_memory = estimate_memory(original_module) # Must improve estimation (accommodate batch sizes etc)

                            # Accommodate on our device if we can
                            if module_memory < self.available_memory:
                                self.available_memory -= module_memory

                            # Distribute otherwise
                            elif module_memory < candidate_node_memory:
                                print(f"distributing {target.attr}")

                                # Wrapping module custom nn.Module that will handle forward and backward passes
                                # between nodes
                                wrapped_module = DistributedModule(self, candidate_node)

                                self.send_module(original_module, candidate_node)
                                setattr(model, target.attr, wrapped_module)
                                candidate_node_memory -= module_memory


# Analyze model for distribution
def distribute_model(model, available_nodes=None, indent=0):
    class Colours:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        RESET = '\033[0m'

    if available_nodes is None:
        # Test set of nodes, real set will be obtained from either the worker's list of nodes,
        # some form of network propagation, or the smart contract
        available_nodes = [
            {"memory": 1e9, "latency_matrix": []},
            {"memory": 1e9, "latency_matrix": []},
            {"memory": 1e9, "latency_matrix": []},
            {"memory": 1e9, "latency_matrix": []}
        ]

    # While we initialize the model candidate worker nodes should be put on standby as we assign the submodules

    # Estimate memory requirements for the model
    model_memory = estimate_memory(model)
    print("   " * indent + f"Parent Module: {round(model_memory / 1e9, 3)} GB")

    # Variables for keeping track of offloaded workers + modules
    candidate_node = max(enumerate([node["memory"] for node in available_nodes]), key=lambda x: x[1])[0]
    indent += 1

    if len(list(model.children())) > 0:
        for name, submodule in model.named_children():
            submodule_memory = estimate_memory(submodule)
            # TODO:
            #  Priority: check for lowest latency x high memory node to offload first submodule to.
            #  Later: if submodule is too big we can call the distribute model again.
            if submodule_memory < available_nodes[candidate_node]["memory"]:
                available_nodes[candidate_node]["memory"] -= submodule_memory
                print(Colours.GREEN + "   " * indent + f"{name}: {round(submodule_memory / 1e9, 3)} GB" + Colours.RESET)
            else:
                available_nodes = available_nodes[:candidate_node] + available_nodes[candidate_node + 1:]
                candidate_node = max(enumerate([node["memory"] for node in available_nodes]), key=lambda x: x[1])[0]
                print("   " * indent + Colours.RED + "Can't accommodate sub-module on worker, distributing further..." +
                      Colours.RESET)
                distribute_model(submodule, available_nodes, indent)


def get_first_layer(model: nn.Module):
    if len(list(model.children())) > 0:
        submodule = next(model.children())
        return get_first_layer(submodule)
    else:
        return model


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
