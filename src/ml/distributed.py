from src.ml.model_analyzer import handle_output, get_first_layer, estimate_memory

import torch.nn as nn
import torch.optim as optim
import torch
import inspect
import time
import ast
import os


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
        memory += 1.4e9

    return memory


class Colours:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


class DistributedModule(nn.Module):
    """
    TODO:
        - Create method for node selection and peers to include in job
    """
    def __init__(self, master_node, worker_node):
        super().__init__()
        self.master_node = master_node
        self.worker_node = worker_node

    def forward(self, *args, **kwargs):
        # Relay forward pass to next node
        self.master_node.send_forward(self.worker_node, (args, kwargs))

        # Store the tensor (will be used for backward pass)
        self.master_node.intermediates[-1].append(handle_output(args))

        time.sleep(1)  # Must find an efficient way to wait for worker response (placeholder is to sleep)

        if not self.master_node.forward_relays.empty():
            # Grab returned tensor
            output = self.master_node.forward_relays.get()

            # Store intermediates and connection for backwards pass
            self.master_node.intermediates.append([self.worker_node, handle_output(output)])

            return output


class DistributedModel(nn.Module):
    def __init__(self, master_node, model, nodes=None):
        super(DistributedModel, self).__init__()
        # self.master_node = master_node

        self.master_node = master_node
        self.model = model
        self.nodes = nodes
        self.graph = []
        self.available_memory = get_gpu_memory()

        # self.distribute_model(self.model)

    def backward(self, loss):

        while len(self.master_node.intermediates) > 0:
            vals = self.master_node.intermediates.pop(-1)

            # Len vals 3 means connection info is present
            if len(vals) == 3:
                connection, assoc_input, assoc_output = vals
                assoc_output.backward(loss, retain_graph=True)
                loss = assoc_input.grad
                self.master_node.send_backward(connection, loss)

                time.sleep(0.5)

                if self.master_node.backward_relays.not_empty:
                    loss = self.master_node.backward_relays.get()
            # Len vals 2 means backwards pass of last or first submodule
            elif len(vals) == 2:
                val1, val2 = vals

                # Pass of the first submodule / section
                if isinstance(val1, torch.Tensor):
                    assoc_input, assoc_output = vals
                    assoc_output.backward(loss, retain_graph=True)
                    time.sleep(0.5)
                    print("BACKWARD DONE!")
                # Pass of the last section
                else:
                    connection, assoc_input = vals

                    # If there are remaining computations between output and last submodule
                    if loss.grad_fn is not None:
                        loss.backward()
                        self.master_node.send_backward(connection, assoc_input.grad)
                    else:
                        self.master_node.send_backward(connection, loss)

                    time.sleep(0.5)

                    if not self.master_node.backward_relays.empty():
                        loss = self.master_node.backward_relays.get()

            else:
                raise "Expect vals to be of length 1, 2 or 3."

        # # Iterate through children in backward pass
        # for device, submodule in reversed(self.graph):
        #     if device is None:
        #         assoc_output = self.master_node.assoc_outputs.get()
        #         assoc_output.backward(loss, retain_graph=True)
        #         loss = self.master_node.assoc_intermediates.get().grad
        #     else:
        #         self.master_node.send_backward(self.nodes, loss)
        #
        #         time.sleep(1)  # Must find an efficient way to wait for worker response (placeholder is to sleep)
        #
        #         if self.master_node.loss_relays.not_empty:
        #             loss = self.master_node.loss_relays.get()

    def distribute_model(self, module, indent=0):
        if self.nodes is None:
            # Test set of nodes, real set will be obtained from either the worker's list of nodes,
            # some form of network propagation, or the smart contract.
            self.nodes = [
                # {"id": 0, "memory": 1.4e9, "connection": None, "latency_matrix": []},
                {"id": 1, "memory": 1.4e9, "connection": self.master_node.outbound[0], "latency_matrix": []}
            ]

        # While we initialize the model candidate worker nodes should be put on standby to receive submodules

        # Estimate memory requirements for the model
        module_memory = estimate_memory(module)
        named_children = list(self.model.children())
        print("   " * indent + f"Parent Module: {round(module_memory / 1e9, 3)} GB")

        # Variables for keeping track of offloaded workers + modules
        candidate_node = max(enumerate([node["memory"] for node in self.nodes]), key=lambda x: x[1])[0]
        indent += 1

        if len(named_children) > 0:
            for name, submodule in module.named_children():
                # TODO:
                #  Priority: check for lowest latency x high memory node to offload first submodule to.
                #  Later: if submodule is too big we can call the distribute model again.
                #  Tail recursion / some output to carry current candidate node. Avoid sending tensor to master if we
                #    handle the next submodule and there isn't any intermediate computations between children.

                submodule_memory = estimate_memory(submodule)

                # Check if best/current candidate can support the submodule
                if submodule_memory < self.nodes[candidate_node]["memory"]:
                    # Update available memory for node
                    self.nodes[candidate_node]["memory"] -= submodule_memory

                    # Add node + submodule to graph
                    self.graph.append((self.nodes[candidate_node]["id"], name, submodule))

                    print(Colours.GREEN + "   " * indent + f"{name}: {round(submodule_memory / 1e9, 3)} GB -> device: "
                                                           f"{self.nodes[candidate_node]['id']}" + Colours.RESET)
                # If we cannot, further distribute the submodule by assigning it to a
                else:
                    self.nodes = self.nodes[:candidate_node] + self.nodes[candidate_node + 1:]
                    candidate_node = max(enumerate([node["memory"] for node in self.nodes]), key=lambda x: x[1])[0]
                    print("   " * indent + Colours.RED + "Cannot accommodate submodule on current worker, "
                                                         "distributing further..." + Colours.RESET)
                    self.distribute_model(submodule, indent)

    def create_distributed_model(self):
        """
        Distribute model to available connected nodes, assign modules based on memory requirements & latency
        """
        # if not self.graph:
        #     self.distribute_model(self.model)

        # Grab model source code
        source_code = inspect.getsource(type(self.model))
        parsed_code = ast.parse(source_code)
        children = dict(self.model.named_children())

        candidate_node = self.nodes[0]
        candidate_node_memory = candidate_node["memory"]

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
                                original_module = getattr(self.model, target.attr)
                                module_memory = estimate_memory(original_module)  # Must improve estimation (accommodate batch sizes etc)

                                # Accommodate on our device if we can
                                if module_memory < self.available_memory:
                                    self.available_memory -= module_memory
                                    self.graph.append((None, target.attr))

                                # Distribute otherwise
                                elif module_memory < candidate_node_memory:
                                    print(f"distributing {target.attr}")

                                    # Wrapping module custom nn.Module that will handle forward and backward passes
                                    # between nodes
                                    wrapped_module = DistributedModule(self.master_node, candidate_node["connection"])

                                    self.master_node.send_module(original_module, candidate_node["connection"])
                                    setattr(self.model, target.attr, wrapped_module)
                                    self.graph.append((candidate_node["connection"], target.attr))
                                    candidate_node_memory -= module_memory
