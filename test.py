from src.ml.worker import Worker
# from src.ml.master import Master
import time


# # List of IP addresses for workers
# worker_ips = ["worker1_ip", "worker2_ip", "worker3_ip"]
#
# init_method = "tcp://" + ",".join(worker_ips) + ":port_number"
#
# dist.init_process_group("gloo", init_method=init_method)
#
#
# def discover_devices():
#     num_cuda_devices = torch.cuda.device_count()
#     devices = []
#
#     for i in range(num_cuda_devices):
#         device_properties = torch.cuda.get_device_properties(i)
#         devices.append({
#             "index": i,
#             "name": device_properties.name,
#             "cuda_cores": device_properties.multi_processor_count * 64  # Assuming 64 CUDA cores per SM
#         })
#
#     return devices
#
#
# def assign_modules(modules, devices, latency_threshold):
#     devices.sort(key=lambda x: x["cuda_cores"], reverse=True)
#
#     assignments = []
#     current_device = 0
#
#     for module in modules:
#         pass


ip = "127.0.0.1"
port = 5026

node = Worker(
    host=ip,
    port=port,
    debug=True
)

node2 = Worker(
    host=ip,
    port=port + 1,
    debug=True
)

master = Master()

node.start()
node2.start()

node.connect_with_node(ip, port + 1)

# with open("tensor.pt", "rb") as f:
#     tensor_bytes = f.read()
# node.send_to_nodes(tensor_bytes)

node.send_to_nodes("sallam".encode())
time.sleep(3)

node2.send_to_nodes("shallom!".encode())
node.stop()
node2.stop()


# from networkx import DiGraph, draw
# import matplotlib.pyplot as plt
# from torchviz import make_dot
#
# from transformers import BertModel, PreTrainedModel
# import torch.nn as nn
# import torch
#
#
# class DirectedGraph:
#     def __init__(self):
#         self.params = {}
#         self.graph = DiGraph()
#         self.optimizer = torch.optim.Adam
#
#     def recurse_model(self, model: nn.Module, input_size: torch.Tensor):
#         if len(list(model.children())) > 0:
#             for name, submodule in model.named_children():
#                 input_size = self.recurse_model(submodule, input_size)
#             return input_size
#         else:
#             # mem_estimate = estimate_memory_requirement(model, input_size, self.optimizer)
#             # print(f"Memory estimate: {mem_estimate}")
#
#             if isinstance(model, nn.modules.sparse.Embedding):
#                 input_size = input_size.size() + (model.embedding_dim,)
#                 print(f"Module Type: {type(model)}, Output size: {input_size}")
#
#             elif hasattr(model, "out_features"):
#                 input_size = input_size.size()[:-1] + (model.out_features,)
#                 print(f"Module Type: {type(model)}, Output size: {input_size}")
#
#             else:
#                 input_size = input_size.size()
#                 print(f"\033[91mModule Type: {type(model)}, Input size: {input_size}\033[0m")
#
#             return torch.zeros(input_size)
#
#     def get_nodes(self, fn):
#         assert not torch.is_tensor(fn)
#
#         if hasattr(fn, "variable"):
#             var = fn.variable
#             self.graph.add_node(size_to_str(var.size()))
#             self.graph.add_edge(size_to_str(var.size()), fn)
#
#         self.graph.add_node(fn)
#
#         if hasattr(fn, "next_functions"):
#             for u in fn.next_functions:
#                 if u[0] is not None:
#                     self.graph.add_edge(u[0], fn)
#                     self.get_nodes(u[0])
#
#         if hasattr(fn, "saved_tensors"):
#             for t in fn.saved_tensors:
#                 self.graph.add_edge(t.name(), fn.name())
#                 self.graph.add_node(t.name())