# from src.ml.worker import Worker
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
# node.start()
# node2.start()
#
# time.sleep(1)
#
# node.connect_with_node(ip, port + 1)
#
# start_time = str(time.time()).encode()
# node.send_to_nodes(start_time)
#
# time.sleep(1)
#
# node.stop()
# node2.stop()
#


from networkx import DiGraph, draw
import matplotlib.pyplot as plt
from torchviz import make_dot

from transformers import BertModel, PreTrainedModel
import torch.nn as nn
import torch


class DDG:
    def __init__(self):
        self.params = {}
        self.seen = set()
        self.graph = DiGraph()

    def get_var_name(self, var):
        # Get the name of the variable and its size
        name = self.params.get(id(var), "")
        return "%s\n %s" % (name, "(" + ", ".join(["%d" % v for v in var.size()]) + ")")

    def get_nodes(self, fn):
        assert not torch.is_tensor(fn)

        print(f"Class: {fn.__class__.__name__}")

        if hasattr(fn, "variable"):
            # If the function has a variable (e.g., layer), add it to the graph
            var = fn.variable
            self.graph.add_node(str(id(var)), label=self.get_var_name(var))
            self.graph.add_edge(str(id(var)), str(id(fn)), label="Layer")

        # Add the current function to the graph
        self.graph.add_node(str(id(fn)), label=str(fn.__class__.__name__))

        if hasattr(fn, "next_functions"):
            for u in fn.next_functions:
                if u[0] is not None:
                    # Add edges and nodes for the next functions in the computation
                    self.graph.add_edge(str(id(u[0])), str(id(fn)), label="Next Function")
                    self.get_nodes(u[0])

        if hasattr(fn, "saved_tensors"):
            for t in fn.saved_tensors:
                # Add edges and nodes for saved tensors
                self.graph.add_edge(str(id(t)), str(id(fn)), label="Saved Tensor")
                self.graph.add_node(str(id(t)), label=self.get_var_name(t))

    def parse_model(self, model: nn.Module, dummy_input: torch.Tensor):
        output = model(dummy_input)
        self.graph = DiGraph()

        # Parse the model to build the graph
        if hasattr(output, "grad_fn"):
            self.get_nodes(output.grad_fn)

        elif hasattr(output, "last_hidden_state"):
            self.get_nodes(output.last_hidden_state.grad_fn)


model = BertModel.from_pretrained('bert-base-uncased')
dummy_input = torch.zeros((1, 3), dtype=torch.long)

for i, (name, submodule) in enumerate(model.named_children()):
    if i == 0:
        dummy_input = submodule(dummy_input)
    else:
        # for n, (name, submodule) in enumerate(submodule.named_children()):
        d = DDG()
        d.parse_model(submodule, dummy_input)

        plt.figure(figsize=(12, 8))
        draw(d.graph, with_labels=True, font_size=8, font_color='black', node_color='skyblue', edge_color='gray', arrowsize=10)
        plt.title("Computation Graph")
        plt.show()
        break
        # break
