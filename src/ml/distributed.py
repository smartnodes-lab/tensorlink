from networkx import DiGraph, draw
import matplotlib.pyplot as plt
from transformers import BertModel
import torch.nn as nn
import torch


def parameter_memory(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


def activation_memory(output: torch.Tensor):
    return output.element_size() * output.numel()


def gradient_memory(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    return sum(g.numel() * g.element_size() for g in gradients) if gradients else 0


def optimizer_memory(optimizer):
    return sum(state.numel() * state.element_size() for group in optimizer.param_groups for state in group['params'])


def estimate_memory_requirement(layer, dummy_input: torch.Tensor, optimizer):
    layer.eval()

    output = layer(dummy_input)
    loss = output.sum()

    optimizer = optimizer(layer.parameters())
    optimizer.zero_grad()
    loss.backward()

    params_mem = parameter_memory(layer)
    activations_mem = activation_memory(output)  # in case of non-modular activations (eg. nn.functional.relu)
    gradient_mem = gradient_memory(layer)
    optimizer_mem = optimizer_memory(optimizer)

    return sum([params_mem, activations_mem, gradient_mem, optimizer_mem]) / (1024 ** 2)


# Attempt to recursively forward pass individual layers to free up memory,
# just testing ideas...
class DirectedGraph:
    def __init__(self):
        self.params = {}
        self.graph = DiGraph()
        self.optimizer = torch.optim.Adam

    def recurse_model(self, model: nn.Module, input_size: torch.Tensor):
        if len(list(model.children())) > 0:
            for name, submodule in model.named_children():
                input_size = self.recurse_model(submodule, input_size)
            return input_size
        else:
            mem_estimate = estimate_memory_requirement(model, input_size, self.optimizer)
            print(f"Memory estimate: {mem_estimate}")

            if isinstance(model, nn.modules.sparse.Embedding):
                input_size = input_size.size() + (model.embedding_dim,)
                print(f"Module Type: {type(model)}, Output size: {input_size}")

            elif hasattr(model, "out_features"):
                input_size = input_size.size()[:-1] + (model.out_features,)
                print(f"Module Type: {type(model)}, Output size: {input_size}")

            else:
                input_size = input_size.size()
                print(f"\033[91mModule Type: {type(model)}, Input size: {input_size}\033[0m")

            return torch.zeros(input_size)


d = DirectedGraph()
m = BertModel.from_pretrained('bert-base-uncased')
out = d.recurse_model(m, torch.zeros((1, 2), dtype=torch.long))


# plt.figure(figsize=(12, 8))
# draw(d.graph, with_labels=True, font_size=8, font_color='black', node_color='skyblue', edge_color='gray', arrowsize=10)
# plt.title("Computation Graph")
# plt.show()
