from src.ml.model_analyzer import handle_output, get_first_layer, estimate_memory, access_module
from src.roles.worker import Worker
from src.p2p.connection import Connection

import torch.nn as nn
import torch.optim as optim
import threading
import inspect
import pickle
import torch
import queue
import time
import ast
import os


MAX_WAIT_TIME = 3


class Trainer:
    def __init__(self, worker):
        self.worker = worker

    def run_training(self):
        pass

    def average_gradients(self):
        pass


class DistributedModel(nn.Module):
    """
    DistributedModel:
        Is able to host offloaded submodules along with local operations. Can be spawned
        by a Worker or a User. Host is referred to as the 'master' node.
    """
    def __init__(self, model: nn.Module, master_node: Worker):
        super(DistributedModel, self).__init__()

        self.master_node = master_node
        self.model = model
        self.model.forward_queues = queue.Queue()
        self.model.backward_queues = queue.Queue()
        self.model.intermediates = queue.LifoQueue()  # Queue to hold intermediates, must be converted into dictionary
                                                      # of queues if we wish to perform multiple epochs concurrently
        self.master_node.modules["Master"] = self.model
        self.spawn_workers = []

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        self.model.intermediates.put(args)
        x = self.model(args)
        return x

    def backward(self, loss):

        # Get the oldest epoch intermediates from storage via dict key lookup (todo)
        while self.model.intermediates.empty() is False:
            vals = self.model.intermediates.get()

            if len(vals) == 3:  # Len == 3 means connection info is present TODO: better context creation
                connection, assoc_input, assoc_output = vals
                assoc_output.backward(loss, retain_graph=True)
                loss = assoc_input.grad
                self.master_node.send_backward(connection, loss)

                if self.model.backward_relays.not_empty:
                    loss = self.model.backward_relays.get()

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

    def distribute_model(self):
        # Retrieve model names and assign workers to offload. Contact candidate workers
        # and ensure they are ready to receive the model / train
        distributed_module_ids = []

        def recurse_model(module, nodes, candidate_node=None, mod_id=None):
            # Get module information
            if mod_id is None:
                mod_id = []

            module_memory = estimate_memory(module)
            module_children = list(module.named_children())
            module_name = f"{type(module)}".split(".")[-1].split(">")[0][:-1]

            # See if we can handle the input first
            if module_memory < self.master_node.available_memory:
                self.master_node.available_memory -= module_memory
                return nodes, f"MASTER"

            elif module_memory < max(nodes, key=lambda x: x["memory"])["memory"]:
                candidate_node = max(enumerate(nodes), key=lambda x: x[1]["memory"])[0]
                nodes[candidate_node]["memory"] -= module_memory
                distributed_module_ids.append(mod_id)
                self.wrap_module(mod_id, nodes[candidate_node]["connection"])
                return nodes, f"{nodes[candidate_node]['id']}"

            # We cannot handle the module on the master node, there are now three options:
            # assume intermediate DistributedModel on master if graph is empty (or user wants to maximize his compute
            # contribution before offloading?)
            # attempt to offload module on best candidate
            # attempt intermediate DistributedModel on best candidate
            # elif not distributed_graph: # Commented out for the same reason as the multi-masternode stuff below...
            # For now, we are just attempting candidate node offload and then defaulting to User masternode distribution
            elif isinstance(module, nn.ModuleList):
                graph = []

                for i, layer in enumerate(module):
                    sub_mod_id = mod_id + [i]

                    # if layer_memory < self.master_node.available_memory:
                    #     self.master_node.available_memory -= layer_memory
                    # elif layer_memory < nodes[candidate_node]["memory"]:
                        # self.wrap_module(sub_mod_id, nodes[candidate_node]["connection"])

                    nodes, subgraph = recurse_model(layer, nodes, candidate_node, sub_mod_id)
                    graph.append(subgraph)
                return nodes, graph

            else:
                # Spawn secondary DistributedModel on master (subgraph)
                graph_name = f"DistributedModel:{module_name}:MASTER"
                graph = {graph_name: {}}

                for i, (name, submodule) in enumerate(module_children):
                    if len(graph[graph_name]) > 0:
                        candidate_node = max(enumerate(nodes), key=lambda x: x[1]["memory"])[0]

                    # Handle submodule
                    nodes, subgraph = recurse_model(submodule, nodes, candidate_node, mod_id + [i])
                    graph[graph_name][f"DistributedSubModule:{name}"] = subgraph

                return nodes, graph

        # Get list of workers and request them be put in a "stand-by" state for the particular request
        # Before we call this method we must:
        #   confirm connecting users via SC if the user is the master
        #   If the worker is the master, residual workers can confirm sub-master via SC seed nodes
        #   if we are in the validator state

        worker_nodes = list(self.master_node.nodes.values())
        nodes, graph = recurse_model(self.model, worker_nodes)

        for thread in self.spawn_workers:
            thread.join()

        return nodes, graph

    # def offload_module(self, parent_module, child_module, connection):
    #     # Wait for some sort of signal from worker to know that it has taken
    #     start_time = time.time()
    #
    #     while True:
    #         if time.time() - start_time > MAX_WAIT_TIME:
    #             # Select another worker if current worker takes too long
    #             connection = max(self.master_node.nodes, key=lambda x: x["memory"])
    #         elif self.master_node

    def wrap_module(self, module_id: list, worker: Connection):
        child_module = access_module(self.model, module_id)
        parent_module = access_module(self.model, module_id[:-1]) if len(module_id) > 1 else self.model

        offloaded_module = OffloadedModule(self.master_node, worker)

        if isinstance(parent_module, nn.ModuleList):
            parent_module[module_id[-1]] = offloaded_module
        else:
            child_name = list(parent_module.named_children())[module_id[-1]][0]
            setattr(parent_module, child_name, offloaded_module)

        # Spawn a worker thread for the offloaded module
        offloaded_module.spawn_worker(child_module, module_id)

        # spawn_worker = threading.Thread(target=offloaded_module.spawn_worker, args=(child_module, module_id))
        # spawn_worker.start()
        # self.spawn_workers.append(spawn_worker)4


class OffloadedModule(nn.Module):
    """
    OffloadedModule:
        A module wrapper that handles offloading on the master node without disrupting normal
        pytorch model flow.
    """
    def __init__(self, master_node: Worker, worker_node: Connection):
        super(OffloadedModule, self).__init__()

        self.master_node = master_node
        self.worker_node = worker_node

        self.module_id = None

    def spawn_worker(self, module: nn.Module, module_id: list):
        # # Initialize a threading Timer to monitor the loading process
        # timer = threading.Timer(MAX_WAIT_TIME, self.handle_timeout)
        # timer.start()

        # try:
        # Send the module to the worker node
        self.module_id = str(module_id).encode()
        module.id = self.module_id

        self.master_node.distributed_graph[self.module_id] = None

        self.master_node.send_module(module, self.worker_node)

        # Wait for the worker node to confirm module loading
        while self.master_node.distributed_graph[self.module_id] is None:
            time.sleep(0.01)

        #     # Module loaded successfully
        # except TimeoutError:
        #     # Timeout occurred, handle it
        #     self.handle_timeout()
        #
        # finally:
        #     # Stop the timer as loading completed (or failed)
        #     timer.cancel()

    def handle_timeout(self):
        # Timeout occurred, switch to another worker
        new_worker_node = self.master_node.select_candidate_worker()
        if new_worker_node:
            self.worker_node = new_worker_node["connection"]
        else:
            # No available workers found, handle the situation accordingly
            pass

    def forward(self, *args, **kwargs):
        # Relay forward pass to next node
        start_time = time.time()
        n_queued = self.master_node.modules["Master"].forward_queues.qsize()  # forward_queues[self.module_id].qsize() TODO, indexing by module id
        self.master_node.send_forward(self.worker_node, (args, kwargs), self.module_id)

        while self.master_node.modules["Master"].forward_queues.qsize() <= n_queued:
            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                pass

        # Grab returned tensor
        output = self.master_node.modules["Master"].forward_queues.get()

        # Store intermediates and connection for backwards pass
        self.master_node.modules["Master"].intermediates.put([self.worker_node, handle_output(output)])

        return output

    # async def async_send_forward(self, node: Connection, args, context: bytes = b""):
    #     pickled_data = b"FORWARD" + context + pickle.dumps(args)
    #
    #     self.send_to_node(node, pickled_data)
    #
    # async def async_send_backward(self, node: Connection, args, context: bytes = b""):
    #     pickled_data = b"BACKWARD" + context + pickle.dumps(args)
    #     self.send_to_node(node, pickled_data)


# class DistributedSubModule(nn.Module):
#     """
#     DistributedSubModule:
#         An offloaded submodule assigned by the master node and run on a worker.
#     """
#     def __init__(self, module: nn.Module, module_id: bytes, worker: Worker, offloaded: Connection):
#         super(DistributedSubModule, self).__init__()
#
#         self.module = module
#         self.module_id = module_id
#
#         self.worker = worker
#         self.offloaded = offloaded
#
#         self.forward_relays = queue.Queue()
#         self.backward_relays = queue.Queue()
#         self.intermediates = queue.Queue()
#
#         self.optimizer = None
#         self.loss = None
#
#     def forward(self):
#         if self.forward_relays.empty() is False:
#             # Grab queued forward pass and unpack values if any
#             prev_forward = self.forward_relays.get()
#             if isinstance(prev_forward, tuple):
#                 args, kwargs = prev_forward
#             else:
#                 args = prev_forward
#                 kwargs = {}
#
#             # Clear tensor of any previous info, set up for custom backward pass
#             _input = handle_output(args).clone().detach().requires_grad_()  # Move the detaching to the end so we don't
#             _output = self.module(_input, **kwargs)                         # send the useless data over p2p
#
#             # Store output and input tensor for backward pass
#             self.intermediates.put([_input, handle_output(_output)])
#
#             # Relay forward pass to the next node
#             self.worker.send_forward(self.offloaded, args, self.module_id)
#
#     def backward(self):
#         # Complete any outstanding back propagations
#         if self.backward_relays.empty() is False:
#             # Grab backwards pass from incoming node and our associated input/output from forward pass
#             loss_relay = self.backward_relays.get()
#             assoc_input, assoc_output = self.intermediates.get()
#
#             # Continue backwards pass on our section of model
#             assoc_output.backward(loss_relay, retain_graph=True)  # Do we need this?
#
#             self.optimizer.zero_grad()
#             self.optimizer.step()
#
#             # Pass along backwards pass to next node
#             dvalues = assoc_input.grad
#
#             self.worker.send_backward(self.offloaded, dvalues, self.module_id)
