# from src.ml.worker import Worker
# # from src.ml.master import Master
# import time
#
#
# # # List of IP addresses for workers
# # worker_ips = ["worker1_ip", "worker2_ip", "worker3_ip"]
# #
# # init_method = "tcp://" + ",".join(worker_ips) + ":port_number"
# #
# # dist.init_process_group("gloo", init_method=init_method)
# #
# #
# # def discover_devices():
# #     num_cuda_devices = torch.cuda.device_count()
# #     devices = []
# #
# #     for i in range(num_cuda_devices):
# #         device_properties = torch.cuda.get_device_properties(i)
# #         devices.append({
# #             "index": i,
# #             "name": device_properties.name,
# #             "cuda_cores": device_properties.multi_processor_count * 64  # Assuming 64 CUDA cores per SM
# #         })
# #
# #     return devices
# #
# #
# # def assign_modules(modules, devices, latency_threshold):
# #     devices.sort(key=lambda x: x["cuda_cores"], reverse=True)
# #
# #     assignments = []
# #     current_device = 0
# #
# #     for module in modules:
# #         pass
#
#
# ip = "127.0.0.1"
# port = 5026
#
# node = Worker(
#     host=ip,
#     port=port,
#     debug=True
# )
#
# node2 = Worker(
#     host=ip,
#     port=port + 1,
#     debug=True
# )
#
# # master = Master()
#
# node.start()
# node2.start()
#
# node.connect_with_node(ip, port + 1)
#
# # with open("tensor.pt", "rb") as f:
# #     tensor_bytes = f.read()
# # node.send_to_nodes(tensor_bytes)
#
# node.send_to_nodes("sallam".encode())
# time.sleep(3)
#
# node2.send_to_nodes("shallom!".encode())
# node.stop()
# node2.stop()


from networkx import DiGraph, draw
import matplotlib.pyplot as plt
from torchviz import make_dot

from transformers import BertModel, PreTrainedModel
import torch.nn as nn
import torch


def recurse_model(model: nn.Module, dummy_input: torch.Tensor):
    # Check for submodules within
    if len(list(model.children())) > 0:
        # Forward pass thru submodels
        if isinstance(model, nn.ModuleList):
            for submodel in list(model.children()):
                if isinstance(dummy_input, tuple):
                    dummy_input = dummy_input[0]
                dummy_input = submodel(dummy_input)
        elif all([len(list(n.children())) == 0 for n in list(model.children())]):
            dummy_input = model(dummy_input)
            return dummy_input
        else:
            for submodule in model.children():
                dummy_input = recurse_model(submodule, dummy_input)
            return dummy_input
    else:
        return model(dummy_input)


def get_memory(module: nn.Module, acc=0):
    if len(list(module.children())) > 0:
        for submodule in module.children():
            acc = get_memory(submodule, acc)
    else:
        mem = sum(p.numel() * p.element_size() for p in module.parameters())
        acc += mem
    return acc


def is_subgraph(onnx_graph, onnx_subgraph):
    # Check if all nodes of the subgraph are in the larger graph
    subgraph_nodes = set(onnx_subgraph.graph.node)
    if not subgraph_nodes.issubset(node.name for node in onnx_graph.graph.node):
        return False

    # Check if all edges of the subgraph are in the larger graph
    subgraph_edges = set((edge.input[0], edge.output[0]) for edge in onnx_subgraph.graph.edge)
    if not subgraph_edges.issubset((edge.input[0], edge.output[0]) for edge in onnx_graph.graph.edge):
        return False

    return True


def contains_residuals(onnx_graph):
    pass


def find_subgraph_location(dag, subgraph):
    def dfs(node, path):
        path.append(node)

        if is_subgraph():
            pass


def get_connections_into_subgraph(graph, subgraph_input_nodes):
    connections = []

    # Traverse inward from input nodes
    for input_node in subgraph_input_nodes:
        connections.extend(traverse_inward(graph, input_node))

    return connections


def traverse_inward(graph, node):
    connections = []

    # Traverse backward along incoming edges
    for edge in graph.edges:
        if edge.target == node:
            connections.append((edge.source, edge.target))
            connections.extend(traverse_inward(graph, edge.source))

    return connections


class DirectedGraph:
    def __init__(self):
        self.params = {}
        self.graph = DiGraph()
        self.optimizer = torch.optim.Adam

            # mem_estimate = estimate_memory_requirement(model, input_size, self.optimizer)
            # print(f"Memory estimate: {mem_estimate}")

            # if isinstance(model, nn.modules.sparse.Embedding):
            #     input_size = input_size.size() + (model.embedding_dim,)
            #     print(f"Module Type: {type(model)}, Output size: {input_size}")
            #
            # elif hasattr(model, "out_features"):
            #     input_size = input_size.size()[:-1] + (model.out_features,)
            #     print(f"Module Type: {type(model)}, Output size: {input_size}")
            #
            # else:
            #     input_size = input_size.size()
            #     print(f"\033[91mModule Type: {type(model)}, Input size: {input_size}\033[0m")

    # def get_nodes(self, fn):
    #     assert not torch.is_tensor(fn)
    #
    #     if hasattr(fn, "variable"):
    #         var = fn.variable
    #         self.graph.add_node(size_to_str(var.size()))
    #         self.graph.add_edge(size_to_str(var.size()), fn)
    #
    #     self.graph.add_node(fn)
    #
    #     if hasattr(fn, "next_functions"):
    #         for u in fn.next_functions:
    #             if u[0] is not None:
    #                 self.graph.add_edge(u[0], fn)
    #                 self.get_nodes(u[0])
    #
    #     if hasattr(fn, "saved_tensors"):
    #         for t in fn.saved_tensors:
    #             self.graph.add_edge(t.name(), fn.name())
    #             self.graph.add_node(t.name())


model = BertModel.from_pretrained('bert-base-uncased')
dummy_in = torch.zeros((1, 1), dtype=torch.long)


mock_peers = [
    {
        "devices": [],
        "latency": [],
        "reputation": []
    },
    {
        "devices": [],
        "latency": [],
        "reputation": []
    },
    {
        "devices": [],
        "latency": [],
        "reputation": []
    },
]

# Get submodules
submodules = list(model.children())

embeddings = submodules[0]
out = embeddings(dummy_in)
dot = make_dot(out, params=dict(list(embeddings.named_parameters())))
dot.render("embeddings.pdf")

encoder = submodules[1]
heads = list(next(encoder.children()).children())
out = heads[0](out.detach())
dot = make_dot(out[0], params=dict(list(heads[0].named_parameters())))
dot.render("head.pdf")

# d = DirectedGraph()
# d.recurse_model(m, dummy_in)
