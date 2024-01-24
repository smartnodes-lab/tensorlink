from transformers import BertModel, AutoModelForCausalLM
from torchviz import make_dot
import torch.nn as nn
import networkx as nx
import torch
import random
import re


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


def handle_output(tensor):
    if hasattr(tensor, "last_hidden_state"):
        tensor = tensor.last_hidden_state
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    return tensor


def create_graph(module: nn.Module, dummy_input: torch.Tensor):
    out = handle_output(module(dummy_input))

    dot = make_dot(out, params=dict(list(module.named_parameters())))
    dot.render(str(round(random.random(), 2)))

    dot_string = dot.source
    lines = dot_string.replace("\n", "").split("\t")
    nodes = [line.strip() for line in lines if "label=" in line]
    nodes = [parse_node(node) for node in nodes]
    nodes = {node[0]: node[1] for node in nodes}
    edges = [line.strip() for line in lines if "->" in line]
    edges = [parse_edge(edge) for edge in edges]

    return nodes, edges


class DAG:
    def __init__(self, model, dummy_input):
        # Model variables
        self.model = model
        nodes, edges = create_graph(model, dummy_input)

        # Model graph variables
        self.nodes = nodes
        self.edges = edges
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edges)

    def is_subgraph(self, subgraph: nx.DiGraph):
        parent_graph = nx.relabel_nodes(self.graph, self.nodes)
        gm = nx.isomorphism.GraphMatcher(parent_graph, subgraph)
        return gm.subgraph_is_monomorphic()


def estimate_memory(module):
    """
    Dummy estimate compared to estimate_memory_requirements but doesn't require a dummy
    forward pass and thus is preferred for now.
    """
    return 4 * sum(param.numel() * param.element_size() for param in module.parameters())


# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
model = BertModel.from_pretrained('bert-base-uncased')
dummy_in = torch.zeros((1, 1), dtype=torch.long)
dag = DAG(model, dummy_in)
