from src.ml.worker import Worker

from transformers import BertModel
import torch
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


ip = "127.0.0.1"
port = 5026


worker1 = Worker(
    host=ip,
    port=port,
    debug=True
)

worker2 = Worker(
    host=ip,
    port=port + 1,
    debug=True
)


worker1.start()
worker2.start()

worker1.connect_with_node("127.0.0.1", port + 1)
time.sleep(1)

worker2.training = True

# print(f"Worker 1: {worker1.all_nodes}")
# print(f"Worker 2: {worker2.all_nodes}")
#
# print("Sending data to Worker 2")
# worker1.send_to_node(worker1.all_nodes[0], b"SALLLAM" * 5_000 + b"SEEEEELIIIIM!")
# time.sleep(1)
#
# worker2.send_to_node(worker2.all_nodes[0], b"Okay.")
# time.sleep(1)


# Worker 1 acts as master node in this scenario
model = BertModel.from_pretrained("bert-base-uncased")
dummy_input = torch.zeros((1, 1), dtype=torch.long)

optimizer = torch.optim.Adam
worker1.distribute_submodules(model)
worker1.model(dummy_input)

# worker1.stop()
# worker2.stop()


# def is_subgraph(onnx_graph, onnx_subgraph):
#     # Check if all nodes of the subgraph are in the larger graph
#     subgraph_nodes = set(onnx_subgraph.graph.node)
#     if not subgraph_nodes.issubset(node.name for node in onnx_graph.graph.node):
#         return False
#
#     # Check if all edges of the subgraph are in the larger graph
#     subgraph_edges = set((edge.input[0], edge.output[0]) for edge in onnx_subgraph.graph.edge)
#     if not subgraph_edges.issubset((edge.input[0], edge.output[0]) for edge in onnx_graph.graph.edge):
#         return False
#
#     return True
#
#
# def contains_residuals(onnx_graph):
#     pass
#
#
# def find_subgraph_location(dag, subgraph):
#     def dfs(node, path):
#         path.append(node)
#
#         if is_subgraph():
#             pass
#
#
# def get_connections_into_subgraph (graph, subgraph_input_nodes):
#     connections = []
#
#     # Traverse inward from input nodes
#     for input_node in subgraph_input_nodes:
#         connections.extend(traverse_inward(graph, input_node))
#
#     return connections
#
#
# def traverse_inward(graph, node):
#     connections = []
#
#     # Traverse backward along incoming edges
#     for edge in graph.edges:
#         if edge.target == node:
#             connections.append((edge.source, edge.target))
#             connections.extend(traverse_inward(graph, edge.source))
#
#     return connections
#
# # mem_estimate = estimate_memory_requirement(model, input_size, self.optimizer)
#     # print(f"Memory estimate: {mem_estimate}")
#
#     # if isinstance(model, nn.modules.sparse.Embedding):
#     #     input_size = input_size.size() + (model.embedding_dim,)
#     #     print(f"Module Type: {type(model)}, Output size: {input_size}")
#     #
#     # elif hasattr(model, "out_features"):
#     #     input_size = input_size.size()[:-1] + (model.out_features,)
#     #     print(f"Module Type: {type(model)}, Output size: {input_size}")
#     #
#     # else:
#     #     input_size = input_size.size()
#     #     print(f"\033[91mModule Type: {type(model)}, Input size: {input_size}\033[0m")
#
#     # def get_nodes(self, fn):
#     #     assert not torch.is_tensor(fn)
#     #
#     #     if hasattr(fn, "variable"):
#     #         var = fn.variable
#     #         self.graph.add_node(size_to_str(var.size()))
#     #         self.graph.add_edge(size_to_str(var.size()), fn)
#     #
#     #     self.graph.add_node(fn)
#     #
#     #     if hasattr(fn, "next_functions"):
#     #         for u in fn.next_functions:
#     #             if u[0] is not None:
#     #                 self.graph.add_edge(u[0], fn)
#     #                 self.get_nodes(u[0])
#     #
#     #     if hasattr(fn, "saved_tensors"):
#     #         for t in fn.saved_tensors:
#     #             self.graph.add_edge(t.name(), fn.name())
#     #             self.graph.add_node(t.name())
