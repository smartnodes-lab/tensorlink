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


def estimate_memory(module):
    """
    Dummy estimate compared to estimate_memory_requirements but doesn't require a dummy
    forward pass and thus is preferred for now.
    """
    return 4 * sum(param.numel() * param.element_size() for param in module.parameters())


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


# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
model = BertModel.from_pretrained('bert-base-uncased')
dummy_in = torch.zeros((1, 1), dtype=torch.long)
dag = DAG(model, dummy_in)

# from networkx import DiGraph, draw
# import matplotlib.pyplot as plt
# from transformers import BertModel
# import torch.nn as nn
# import numpy as np
# import torch
# import re
#
#
# def _node_get(node: torch._C.Node, key: str):
#     """Gets attributes of a node which is polymorphic over return type."""
#     sel = node.kindOf(key)
#     return getattr(node, sel)(key)
#
#
# torch._C.Node.__getitem__ = _node_get
#
#
# THEMES = {
#     "basic": {
#         "background_color": "#FFFFFF",
#         "fill_color": "#E8E8E8",
#         "outline_color": "#000000",
#         "font_color": "#000000",
#         "font_name": "Times",
#         "font_size": "10",
#         "margin": "0,0",
#         "padding":  "1.0,0.5",
#     },
#     "blue": {
#         "background_color": "#FFFFFF",
#         "fill_color": "#BCD6FC",
#         "outline_color": "#7C96BC",
#         "font_color": "#202020",
#         "font_name": "Verdana",
#         "font_size": "10",
#         "margin": "0,0",
#         "padding":  "1.0,0.5",
#     },
# }
#
#
# def size_to_str(size):
#     return '(' + ', '.join(['%d' % v for v in size]) + ')'
#
#
# def get_shape(torch_node):
#     m = re.match(r".*Float\(([\d\s\,]+)\).*", str(next(torch_node.outputs())))
#     if m:
#         shape = m.group(1)
#         shape = shape.split(",")
#         shape = tuple(map(int, shape))
#     else:
#         shape = None
#     return shape
#
#
# def torch_id(node):
#     # op = node.kind()
#     # input_ids = [i.unique()]
#     return node.scopeName() + "/outputs/" + "/".join(["{}".format(o.unique()) for o in node.outputs()])
#
#
# def node_id(node):
#     return node.id if hasattr(node, "id") else hash(node)
#
#
# class Node:
#     """Represents a framework-agnostic neural network layer in a directed graph."""
#
#     def __init__(self, uid, name, op, output_shape=None, params=None):
#         """
#         uid: unique ID for the layer that doesn't repeat in the computation graph.
#         name: Name to display
#         op: Framework-agnostic operation name.
#         """
#         self.id = uid
#         self.name = name  # TODO: clarify the use of op vs name vs title
#         self.op = op
#         self.repeat = 1
#         if output_shape:
#             assert isinstance(output_shape, (tuple, list)),\
#             "output_shape must be a tuple or list but received {}".format(type(output_shape))
#         self.output_shape = output_shape
#         self.params = params if params else {}
#         self._caption = ""
#         self.memory_reqs = None
#
#     @property
#     def title(self):
#         # Default
#         title = self.name or self.op
#
#         if "kernel_shape" in self.params:
#             # Kernel
#             kernel = self.params["kernel_shape"]
#             title += "x".join(map(str, kernel))
#         if "stride" in self.params:
#             stride = self.params["stride"]
#             if np.unique(stride).size == 1:
#                 stride = stride[0]
#             if stride != 1:
#                 title += "/s{}".format(str(stride))
#         #         # Transposed
#         #         if node.transposed:
#         #             name = "Transposed" + name
#         return title
#
#     @property
#     def caption(self):
#         if self._caption:
#             return self._caption
#
#         caption = ""
#
#         # Stride
#         # if "stride" in self.params:
#         #     stride = self.params["stride"]
#         #     if np.unique(stride).size == 1:
#         #         stride = stride[0]
#         #     if stride != 1:
#         #         caption += "/{}".format(str(stride))
#         return caption
#
#     def __repr__(self):
#         args = (self.op, self.name, self.id, self.title, self.repeat)
#         f = "<Node: op: {}, name: {}, id: {}, title: {}, repeat: {}"
#         if self.output_shape:
#             args += (str(self.output_shape),)
#             f += ", shape: {:}"
#         if self.params:
#             args += (str(self.params),)
#             f += ", params: {:}"
#         f += ">"
#         return f.format(*args)
#
#
# # Attempt to recursively forward pass individual layers to free up memory,
# # just testing ideas...
# class DAG2:
#     def __init__(self):
#         self.nodes = {}
#         self.edges = []
#         self.theme = THEMES["basic"]
#
#     def add_node(self, node):
#         id = node_id(node)
#         self.nodes[id] = node
#
#     def add_edge(self, node1, node2, label=None):
#         edge = (node_id(node1), node_id(node2), label)
#         if edge not in self.edges:
#             self.edges.append(edge)
#
#     def add_edge_by_id(self, vid1, vid2, label=None):
#         self.edges.append((vid1, vid2, label))
#
#     def outgoing(self, node):
#         nodes = node if isinstance(node, list) else [node]
#         node_ids = [node_id(n) for n in nodes]
#         outgoing = [self[e[1]] for e in self.edges if e[0] in node_ids
#                     and e[1] not in node_ids]
#         return outgoing
#
#     def incoming(self, node):
#         nodes = node if isinstance(node, list) else [node]
#         node_ids = [node_id(n) for n in nodes]
#         incoming = [self[e[0]] for e in self.edges if e[1] in node_ids
#                     and e[0] not in node_ids]
#         return incoming
#
#     def siblings(self, node):
#         incoming = self.incoming(node)
#         if len(incoming) == 1:
#             incoming = incoming[0]
#             siblings = self.outgoing(incoming)
#             return siblings
#         else:
#             return [node]
#
#     def __getitem__(self, key):
#         if isinstance(key, list):
#             return [self.nodes.get(k) for k in key]
#         else:
#             return self.nodes.get(key)
#
#     def remove(self, nodes):
#         """Remove a node and its edges."""
#         nodes = nodes if isinstance(nodes, list) else [nodes]
#         for node in nodes:
#             k = self.id(node)
#             self.edges = list(filter(lambda e: e[0] != k and e[1] != k, self.edges))
#             del self.nodes[k]
#
#     def replace(self, nodes, node):
#         """Replace nodes with node. Edges incoming to nodes[0] are connected to
#         the new node, and nodes outgoing from nodes[-1] become outgoing from
#         the new node."""
#         nodes = nodes if isinstance(nodes, list) else [nodes]
#         # Is the new node part of the replace nodes (i.e. want to collapse
#         # a group of nodes into one of them)?
#         collapse = node_id(node) in self.nodes
#         # Add new node and edges
#         if not collapse:
#             self.add_node(node)
#         for in_node in self.incoming(nodes):
#             # TODO: check specifically for output_shape is not generic. Consider refactoring.
#             self.add_edge(in_node, node, in_node.output_shape if hasattr(in_node, "output_shape") else None)
#         for out_node in self.outgoing(nodes):
#             self.add_edge(node, out_node, node.output_shape if hasattr(node, "output_shape") else None)
#         # Remove the old nodes
#         for n in nodes:
#             if collapse and n == node:
#                 continue
#             self.remove(n)
#
#     def search(self, pattern):
#         """Searches the graph for a sub-graph that matches the given pattern
#         and returns the first match it finds.
#         """
#         for node in self.nodes.values():
#             match, following = pattern.match(self, node)
#             if match:
#                 return match, following
#         return [], None
#
#     def create_graph(self, model, args):
#         self.nodes = {}
#         self.edges = []
#
#         trace, out = torch.jit._get_trace_graph(model, args)
#         torch_graph = torch.onnx._optimize_graph(trace, torch.onnx.OperatorExportTypes.ONNX)
#
#         for torch_node in torch_graph.nodes():
#             op = torch_node.kind() + str(torch_node.output().type().sizes())
#             params = {k: torch_node[k] for k in torch_node.attributeNames()}
#             outputs = [o.unique() for o in torch_node.outputs()]  # TODO: inputs = [i.unique() for i in node.inputs()]
#
#             # Get output shape
#             output_shape = get_shape(torch_node)
#             node_name = torch_id(torch_node)
#
#             # Add HL node
#             hl_node = Node(uid=node_name, name=None, op=op,
#                            output_shape=output_shape, params=params)
#             # hl_node.memory_reqs = estimate_memory_requirement()
#             self.add_node(hl_node)
#
#             # Add edges
#             for target_torch_node in torch_graph.nodes():
#                 target_inputs = [i.unique() for i in target_torch_node.inputs()]
#                 if set(outputs) & set(target_inputs):
#                     self.add_edge_by_id(torch_id(torch_node), torch_id(target_torch_node), output_shape)
#
#     def build_dot(self, filename):
#         """Generate a GraphViz Dot graph.
#
#         Returns a GraphViz Digraph object.
#         """
#         # Build GraphViz Digraph
#         from graphviz import Digraph
#
#         dot = Digraph()
#         dot.attr("graph",
#                  bgcolor=self.theme["background_color"],
#                  color=self.theme["outline_color"],
#                  fontsize=self.theme["font_size"],
#                  fontcolor=self.theme["font_color"],
#                  fontname=self.theme["font_name"],
#                  margin=self.theme["margin"],
#                  rankdir="TD",
#                  pad=self.theme["padding"])
#         dot.attr("node", shape="box",
#                  style="filled", margin="0,0",
#                  fillcolor=self.theme["fill_color"],
#                  color=self.theme["outline_color"],
#                  fontsize=self.theme["font_size"],
#                  fontcolor=self.theme["font_color"],
#                  fontname=self.theme["font_name"])
#         dot.attr("edge", style="solid",
#                  color=self.theme["outline_color"],
#                  fontsize=self.theme["font_size"],
#                  fontcolor=self.theme["font_color"],
#                  fontname=self.theme["font_name"])
#
#         for k, n in self.nodes.items():
#             label = "<tr><td cellpadding='6'>{}</td></tr>".format(n.title)
#             if n.caption:
#                 label += "<tr><td>{}</td></tr>".format(n.caption)
#             if n.repeat > 1:
#                 label += "<tr><td align='right' cellpadding='2'>x{}</td></tr>".format(n.repeat)
#             label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
#             dot.node(str(k), label)
#
#         for a, b, label in self.edges:
#             if isinstance(label, (list, tuple)):
#                 label = "x".join([str(l or "?") for l in label])
#
#             dot.edge(str(a), str(b), label)
#
#         dot.render(filename=filename)
