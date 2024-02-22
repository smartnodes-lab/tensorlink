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
        memory += 0.5e9

    return memory


class Colours:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    PURPLE = '\033[95m'
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
        # self.event = threading.Event()

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

        self.create_distributed_model(self.model)

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

    def create_distributed_model(self, model):
        """
        Distribute model to available connected nodes, assign modules based on memory requirements & latency.
        Replace distributed modules with shell objects in master's instantiation to preserve tensor-flow
        """

        # Grab model source code
        source_code = inspect.getsource(type(model))
        parsed_code = ast.parse(source_code)
        children = dict(model.named_children())

        candidate_node = self.nodes[0]
        candidate_node_memory = candidate_node["memory"]

        submodule_counter = []

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
                                module_memory = estimate_memory(original_module)  # Must improve estimation (accommodate batch sizes etc)

                                # Accommodate on our device if we can
                                if module_memory < self.available_memory:
                                    self.available_memory -= module_memory
                                    self.graph.append((None, target.attr))

                                # If submodule is a large module list, distribute the contents in a leaf-like structure
                                elif isinstance(original_module, nn.ModuleList):
                                    for name, submodule in original_module.named_children():
                                        submodule_memory = estimate_memory(submodule)
                                        if submodule_memory < self.available_memory:
                                            self.graph.append((None, name))
                                            self.available_memory -= submodule_memory
                                        elif submodule_memory < candidate_node_memory:
                                            print(f"distributing {target.attr}")
                                            self.master_node.send_module(submodule, candidate_node["connection"])
                                            candidate_node_memory -= submodule_memory
                                        else:
                                            self.nodes = self.nodes[:candidate_node["id"]] + self.nodes[
                                                                                             candidate_node["id"] + 1:]
                                            candidate_node = max(self.nodes, key=lambda x: x["memory"])

                                # Distribute otherwise
                                else:
                                    wrapped_module = DistributedModule(self.master_node, candidate_node["connection"])

                                    # Try offloading entire model to worker
                                    if module_memory < candidate_node_memory:
                                        print(f"distributing {target.attr}")
                                        # Wrapping module custom nn.Module that will handle forward and backward passes
                                        # between nodes
                                        self.master_node.send_module(original_module, candidate_node["connection"])
                                        candidate_node_memory -= module_memory

                                    # Assign worker as local master node otherwise
                                    else:
                                        print(f"distributing dist_{target.attr}")
                                        self.master_node.send_module(original_module, candidate_node["connection"], b"D")

                                        self.nodes = self.nodes[:candidate_node["id"]] + self.nodes[candidate_node["id"] + 1:]

                                        candidate_node = max(self.nodes, key=lambda x: x["memory"])

                                    self.graph.append((candidate_node["connection"], target.attr))
                                    setattr(self.model, target.attr, wrapped_module)


# Test simulation of the distribution process
def print_distribute_model(module, nodes=None, candidate_node=None, indent=0):
    if nodes is None:
        # Test set of nodes, real set will be obtained from either the worker's list of nodes,
        # some form of network propagation, or the smart contract.768uy
        nodes = [
            {"id": 0, "memory": 1e9, "connection": 0, "latency_matrix": [], "colour": Colours.GREEN},
            {"id": 1, "memory": 1e9, "connection": 0, "latency_matrix": [], "colour": Colours.RED},
            {"id": 2, "memory": 0.2e9, "connection": 0, "latency_matrix": [], "colour": Colours.YELLOW},
            {"id": 3, "memory": 0.2e9, "connection": 0, "latency_matrix": [], "colour": Colours.BLUE},
            {"id": 4, "memory": 0.5e9, "connection": 0, "latency_matrix": [], "colour": Colours.PURPLE}
        ]

    # While we initialize the model candidate worker nodes should be put on standby to receive submodules

    # Estimate memory requirements for the model
    module_memory = estimate_memory(module)
    module_children = list(module.named_children())
    module_name = f"{type(module)}".split(".")[-1].split(">")[0][:-1]
    prefix = "  " * indent

    if candidate_node is None:
        candidate_node = max(nodes, key=lambda x: x["memory"])

    # See if we can handle module on current node
    if module_memory < candidate_node["memory"]:
        print(candidate_node["colour"] + prefix + f"Loaded: {module_name} on worker: {candidate_node['id']}")
        nodes[candidate_node['id']]["memory"] -= module_memory
        return nodes

    # Workflow for non-modularized modules (ie can't be distributed linearly) [currently just handles for modulelist]
    elif any(isinstance(module_children[i][1], nn.ModuleList) for i in range(len(module_children))):
        # Update candidate node to the best worker to handle the majority of work
        candidate_node = max(nodes, key=lambda x: x["memory"])
        print(candidate_node["colour"] + prefix + f"Loaded Skeleton Module: {module_name} on worker: {candidate_node['id']}")
        print(prefix + f"Distributing Leafs: {module_children[0][0]}")

        # Must dish out submodules in a leaf-like manner not, filling up the master (current) node first
        # The workflow below can be condensed into some sort of recursive call
        for _, parent_submodule in module_children:
            parent_submodule_memory = estimate_memory(parent_submodule)

            # See if parent submodule can be loaded on current node
            if parent_submodule_memory < candidate_node["memory"]:
                nodes = print_distribute_model(parent_submodule, nodes, candidate_node, indent=indent+1)
            # Else it has to be split up,
            else:
                # See if we can distribute parent module submodules
                for name, submodule in parent_submodule.named_children():
                    submodule_memory = estimate_memory(submodule)
                    # Fill up parent if we can (pass candidate node)
                    if submodule_memory < candidate_node["memory"]:
                        nodes = print_distribute_model(submodule, nodes, candidate_node, indent=indent+1)
                    # Distribute otherwise
                    else:
                        nodes = print_distribute_model(submodule, nodes, indent=indent+1)

    # Module is too large but IS modularizable
    else:
        print(candidate_node["colour"] + prefix + f"Distributing {module_name} Modules...")

        # Send submodule
        for name, submodule in module_children:
            nodes = print_distribute_model(submodule, nodes, candidate_node, indent=indent+1)

            # Fill workers up with submodules before switching to the next
            # candidate_node = max(nodes, key=lambda x: x["memory"])

    # TODO:
    #  Priority: check for lowest latency x high memory node to offload first submodule to.
    #  Later: if submodule is too big we can call the distribute model again.
    #  Tail recursion / some output to carry current candidate node. Avoid sending tensor to master if we
    #    handle the next submodule and there isn't any intermediate computations between children.
