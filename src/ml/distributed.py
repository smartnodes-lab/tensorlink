import torch.nn as nn
import torch.optim as optim
import torch
import time
import os


class Colours:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


class DistributedModel(nn.Module):
    def __init__(self, master_node, model, nodes=None):
        super(DistributedModel, self).__init__()
        # self.master_node = master_node

        self.master_node = master_node
        self.model = model
        self.nodes = nodes
        self.graph = {}

        self.distribute_model(self.model)

    def backward(self, loss):
        loss.backward()
        for k, v in reversed(self.graph.items()):
            if k == 1:  # Placeholder for worker ID
                self.master_node.send_backward()

    def distribute_model(self, module, indent=0):
        if self.nodes is None:
            # Test set of nodes, real set will be obtained from either the worker's list of nodes,
            # some form of network propagation, or the smart contract
            self.nodes = [
                {"id": 0, "memory": 1.4e9, "connection": Connection, "latency_matrix": []},
                {"id": 1, "memory": 1.4e9, "latency_matrix": []}
            ]

        # While we initialize the model candidate worker nodes should be put on standby to receive submodules

        # Estimate memory requirements for the model
        module_memory = estimate_memory(module)
        named_children = list(self.model.children())
        print("   " * indent + f"Parent Module: {round(module_memory / 1e9, 3)} GB")

        # Variables for keeping track of offloaded workers + modules
        candidate_node = max(enumerate([node["memory"] for node in self.nodes]), key=lambda x: x[1])[0]
        indent += 1

        # if len(named_children) == 1:
        #     name = named_children[0][0]
        #     if module_memory < self.nodes[candidate_node]["memory"]:
        #         # Update available memory for node
        #         self.nodes[candidate_node]["memory"] -= module_memory
        #
        #         # Add node + submodule to graph
        #         self.graph[name] = self.nodes[candidate_node]["id"]
        #
        #         print(Colours.GREEN + "   " * indent + f"{name}: {round(module_memory / 1e9, 3)} GB -> device: "
        #                                                f"{self.nodes[candidate_node]['id']}" + Colours.RESET)

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
                    self.graph[name] = self.nodes[candidate_node]["id"]

                    print(Colours.GREEN + "   " * indent + f"{name}: {round(submodule_memory / 1e9, 3)} GB -> device: "
                                                           f"{self.nodes[candidate_node]['id']}" + Colours.RESET)
                else:
                    self.nodes = self.nodes[:candidate_node] + self.nodes[candidate_node + 1:]
                    candidate_node = max(enumerate([node["memory"] for node in self.nodes]), key=lambda x: x[1])[0]
                    print("   " * indent + Colours.RED + "Can't accommodate sub-module on worker, distributing further..." +
                          Colours.RESET)
                    self.distribute_model(submodule, indent)


def estimate_memory(module):
    """
    Dummy estimate compared to estimate_memory_requirements but doesn't require a dummy
    forward pass and thus is preferred for now.
    """
    return 4 * sum(param.numel() * param.element_size() for param in module.parameters())
