from src.ml.model_analyzer import *
from src.p2p.torch_node import TorchNode
from src.p2p.connection import Connection

from torch.nn import Parameter
from typing import Iterator
import torch.nn as nn
import torch.optim as optim
from types import GeneratorType
import threading
import torch
import queue
import time


MAX_WAIT_TIME = 3
THREAD_STORAGE = threading.local()


def contains_offloaded(module: nn.Module):
    if not list(module.named_children()):
        return False
    children = list(module.children())
    exists = False
    for child in children:
        # check if insntance offloadedsubmodule
        if isinstance(child, OffloadedModule):
            return True
        exists = exists or contains_offloaded(child)
    return exists


class DistributedModel(nn.Module):
    """
    DistributedModel:
        Is able to host offloaded submodules along with local operations. Can be spawned
        by a Worker or a User. Host is referred to as the 'master' node.
    """

    def __init__(
        self,
        model: nn.Module,
        master_node: TorchNode,
        batch_size: int,
        micro_batch_size: int,
        config=None,
        device=None,
    ):
        super(DistributedModel, self).__init__()

        self.master_node = master_node
        self.model = model

        assert (
            batch_size % micro_batch_size == 0
        ), "Micro-batch must be divisible by batch."

        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size

        self.model.n_batch = 0

        self.model.forward_queues = {}
        self.model.backward_queues = {}
        self.model.intermediates = [
            []
        ]  # Queue to hold intermediates, must be converted into dictionary
        # of queues if we wish to perform multiple epochs concurrently
        self.master_node.modules["Master"] = self.model

        self.graph = self.distribute_model(config)
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.graph)

        self.spawn_workers = []

    def forward(self, *args, **kwargs):
        """
        Performs the forward pass through the model.
        - Splits input into micro-batches and runs them in parallel.
        - TODO Creates multiple parallel streams of workers for model parallel acceleration
        """
        if len(args) == 1:
            args = args[0]

        n_micro_batches = self.batch_size // self.micro_batch_size

        # Split the input tensor into micro-batches
        micro_batches = torch.chunk(args, n_micro_batches)

        # Create queues for forward and backward passes
        self.model.forward_queues = {m: queue.Queue() for m in range(n_micro_batches)}
        self.model.backward_queues = {m: queue.Queue() for m in range(n_micro_batches)}
        self.model.outputs = {}
        self.model.intermediates = {m: [[]] for m in range(n_micro_batches)}

        # Spawn separate threads for each micro batch for pipeline parallelism
        threads = []
        for micro, batch in enumerate(micro_batches):
            t = threading.Thread(
                target=self.perform_micro_forward, args=(micro, batch), kwargs=kwargs
            )
            t.start()
            threads.append(t)
            time.sleep(0.5)

        for t in threads:
            t.join()

        return self.model.outputs

    def backward(self, loss):
        """
        Performs the backward pass through the model.
        - Splits input into micro-batches and runs them in parallel.
        - TODO Creates multiple parallel streams of workers for model parallel acceleration
        - TODO halt worker parameter updates until final micro-batch
        """
        n_micro_batches = self.batch_size // self.micro_batch_size

        threads = []
        for micro in range(n_micro_batches):
            t = threading.Thread(
                target=self.perform_micro_backward, args=(micro, loss[micro])
            )
            t.start()
            threads.append(t)
            time.sleep(0.5)

        for t in threads:
            t.join()

    def perform_micro_backward(self, micro, loss):
        """"""
        THREAD_STORAGE.micro = micro

        # Get the oldest epoch intermediates from storage via dict key lookup (todo)
        while len(self.model.intermediates[micro]) > 0:
            vals = self.model.intermediates[micro].pop(-1)

            if (
                len(vals) == 3
            ):  # Len == 3 means connection info is present TODO: better context creation
                assoc_input, assoc_output, module_id = vals
                tag = [self.model.n_batch, micro, module_id]

                assoc_output.backward(loss, retain_graph=True)
                loss = assoc_input.grad
                connection = self.master_node.distributed_graph[module_id]

                self.master_node.send_backward(connection, loss, context=tag)

                if self.model.backward_relays.not_empty:
                    loss = self.model.backward_relays.get()

            # Len vals 2 means backwards pass of last submodule
            elif len(vals) == 2:
                val1, val2 = vals

                # Pass of the first submodule / section contains tensor in the first position
                if isinstance(val1, torch.Tensor):
                    assoc_output, _ = vals
                    loss = assoc_output.backward(loss, retain_graph=True)
                # Pass of the last section
                else:
                    module_id, assoc_input = vals
                    tag = [self.model.n_batch, micro, module_id]

                    connection = self.master_node.distributed_graph[module_id]

                    # If there are remaining computations between output and last submodule
                    start_time = time.time()
                    n_queued = self.model.backward_queues[micro].qsize()

                    if loss.grad_fn is not None:
                        loss.backward()
                        self.master_node.send_backward(
                            connection, assoc_input.grad, context=tag
                        )
                    else:
                        self.master_node.send_backward(connection, loss, context=tag)

                    while not self.model.backward_queues[micro].qsize() <= n_queued:
                        if time.time() - start_time >= MAX_WAIT_TIME:
                            # Logic here to request another worker take his place
                            pass

                    loss = self.model.backward_queues[micro].get()[-1]

            elif len(vals) == 1:
                assoc_input = vals[0]
                if isinstance(assoc_input, torch.Tensor):
                    assoc_input.backward(loss)
            else:
                raise "Expect vals to be of length 2 or 3."

    def perform_micro_forward(self, micro, *args, **kwargs):
        THREAD_STORAGE.micro = micro
        x = self.model(*args, **kwargs)
        self.model.outputs[micro] = x

    def train(self, mode: bool = True):
        """
        Sets the training mode for the model and sends requests to offloaded modules to update their training state.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        threads = []

        def recurse_modules(module):
            for children in module.children():
                if isinstance(children, OffloadedModule):
                    self.master_node.send_update_train_request(
                        children.worker_node, mode, children.module_id
                    )
                    t = threading.Thread(
                        target=self.wait_for_state_update,
                        args=(children.module_id, "train"),
                    )
                    t.start()
                    threads.append(t)
                elif contains_offloaded(children):
                    recurse_modules(children)

        recurse_modules(self.model)

        for thread in threads:
            thread.join()

    def eval(self):
        self.train(False)

    def parameters(self, recurse: bool = True, distributed: bool = False):
        """
        Collects parameters from all modules (including offloaded) asynchronously and returns them.
        """
        if distributed:
            parameters = []
            parameter_requests = []

            def recurse_parameters(module):
                for children in module.children():
                    if isinstance(children, OffloadedModule):
                        self.master_node.send_parameters_req(
                            children.worker_node, children.module_id
                        )
                        t = threading.Thread(
                            target=self.wait_for_parameters, args=(children.module_id,)
                        )
                        t.start()
                        parameter_requests.append(t)
                        parameters.append(children.module_id)
                    elif contains_offloaded(children):
                        recurse_parameters(children)
                    else:
                        parameters.append(children.parameters())

            recurse_parameters(self.model)

            for req in parameter_requests:
                req.join()

            for i in range(len(parameters)):
                if isinstance(parameters[i], GeneratorType):
                    pass
                else:
                    params = self.master_node.parameters[parameters[i]]
                    parameters[i] = params

        else:
            parameters = self.model.parameters()

        return parameters

    def wait_for_parameters(self, module_id: bytes):
        """
        Waits until parameters have been received from the specified offloaded module
        """
        while module_id not in self.master_node.parameters.keys():
            # if timeout > time.time() - start_time
            pass

    def wait_for_state_update(self, module_id, state):
        """
        Waits until a state update has been confirmed from the specified offloaded module
        """
        while module_id not in self.master_node.state_updates.keys():
            # if timeout > time.time() - start_time
            pass

        while state not in self.master_node.state_updates[module_id].keys():
            # if timeout > time.time() - start_time
            pass

    def get_node_most_memory(self):
        # gets node with most memory, returns key and memory
        key = max(
            self.master_node.nodes, key=lambda x: self.master_node.nodes[x]["memory"]
        )
        return key, self.master_node.nodes[key]["memory"]

    def distribute_model(self, config=None):
        # Retrieve model names and assign workers to offload. Contact candidate workers
        # and ensure they are ready to receive the model / train
        distributed_module_ids = []

        if config:

            for mod_name, worker_id in config.items():
                node = self.master_node.nodes[worker_id]
                target = find_module(self.model, mod_name)
                assert target is not None, f"Module {mod_name} not found in model."
                module, mod_ids = target  # unpacking the tuple
                self.wrap_module(mod_ids, node)
                return 0, 0

        def recurse_model(module, mod_id=[]):
            # Get module information
            module_memory = estimate_memory(module)
            module_children = list(module.named_children())
            module_type = f"{type(module)}".split(".")[-1].split(">")[0][:-1]

            # See if we can handle the input first
            if module_memory < self.master_node.available_memory:
                self.master_node.available_memory -= module_memory
                return [
                    {
                        "id": "MASTER",
                        "type": module_type,
                        "module_id": mod_id,
                    }
                ]

            elif module_memory < self.get_node_most_memory()[1]:
                # worker_key = self.get_node_most_memory()[0]
                # self.master_node.nodes[worker_key]["memory"] -= module_memory
                distributed_module_ids.append(mod_id)
                self.wrap_module(mod_id, self.master_node.nodes[worker_key])
                return [
                    {
                        "id": worker_key,
                        "type": module_type,
                        "module_id": mod_id,
                    }
                ]

            # We cannot handle the module on the master node, there are now three options:
            # assume intermediate DistributedModel on master if graph is empty (or user wants to maximize his compute
            # contribution before offloading?)
            # attempt to offload module on best candidate
            # attempt intermediate DistributedModel on best candidate
            # elif not distributed_graph: # Commented out for the same reason as the multi-masternode stuff below...
            # For now, we are just attempting candidate node offload and then defaulting to User masternode distribution
            else:
                # Spawn secondary DistributedModel on master (subgraph)
                graph = []

                for i, (name, submodule) in enumerate(module_children):
                    sub_mod_id = mod_id + [i]

                    # Handle submodule
                    subgraph = recurse_model(submodule, sub_mod_id)
                    graph.extend(subgraph)

                return graph

        # Get list of workers and request them be put in a "stand-by" state for the particular request
        # Before we call this method we must:
        #   confirm connecting users via SC if the user is the master
        #   If the worker is the master, residual workers can confirm sub-master via SC seed nodes
        #   if we are in the validator state

        graph = recurse_model(self.model)

        # for thread in self.spawn_workers:
        #     thread.join()

        return graph

    def wrap_module(self, module_id: list, worker: Connection):
        child_module, module_name = access_module(self.model, module_id)
        parent_module = (
            access_module(self.model, module_id[:-1])[0]
            if len(module_id) > 1
            else self.model
        )

        offloaded_module = OffloadedModule(self.master_node, worker)

        # Remove offloaded module from main model optimizer
        child_params = set(
            child_module.parameters()
        )  # Using set for efficient membership check
        current_params = {name: param for name, param in self.named_parameters()}

        # Remove the parameters of the child_module from the current parameters
        for name, param in list(current_params.items()):
            if param in child_params:
                del self.state_dict()[name]

        # Load the updated parameters into the model
        # self.load_state_dict(current_params)

        if isinstance(parent_module, nn.ModuleList):
            parent_module[module_id[-1]] = offloaded_module
        else:
            child_name = list(parent_module.named_children())[module_id[-1]][0]
            setattr(parent_module, child_name, offloaded_module)

        # Spawn a worker thread for the offloaded module
        offloaded_module.spawn_worker(child_module, module_id)

        # spawn_worker = threading.Thread(target=offloaded_module.spawn_worker, args=(child_module, module_id))
        # spawn_worker.start()
        # self.spawn_workers.append(spawn_worker)


class OffloadedModule(nn.Module):
    """
    OffloadedModule:
        A module wrapper that handles offloading on the master node without disrupting normal
        pytorch model flow.
    """

    def __init__(self, master_node: TorchNode, worker_node: Connection):
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
        start_time = time.time()
        n_iter = self.master_node.modules["Master"].n_batch
        n_micro = getattr(THREAD_STORAGE, "micro", None)
        n_queued = self.master_node.modules["Master"].forward_queues[n_micro].qsize()

        tag = [n_iter, n_micro, self.module_id]

        # Store the intermediate tensor for backwards pass
        self.master_node.modules["Master"].intermediates[n_micro][-1].extend(
            [handle_output(args), self.module_id]
        )

        # Relay forward pass to next node
        self.master_node.send_forward(self.worker_node, (args, kwargs), context=tag)

        # Wait for response, change to appending waiting thread to list in master
        while (
            self.master_node.modules["Master"].forward_queues[n_micro].qsize()
            <= n_queued
        ):
            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                pass

        # Grab returned tensor
        _, output = self.master_node.modules["Master"].forward_queues[n_micro].get()

        inter_storage = [
            self.module_id,
            handle_output(output),
        ]  # Store associated output

        # Store intermediates and connection for backwards pass
        self.master_node.modules["Master"].intermediates[n_micro].append(inter_storage)

        return output
