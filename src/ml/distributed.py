from src.ml.model_analyzer import *
from src.p2p.connection import Connection

from transformers.utils import ModelOutput
from torch.nn import Parameter
import torch.optim as optim
import torch.nn as nn
import torch

from multiprocessing import shared_memory
from collections import defaultdict
from types import GeneratorType
from typing import Iterator
from copy import deepcopy
import threading
import hashlib
import pickle
import random
import queue
import time
import json


MAX_WAIT_TIME = 300
THREAD_STORAGE = threading.local()


def get_from_shared_memory(size, name):
    shm = shared_memory.SharedMemory(name=name)
    buffer = shm.buf[:size]
    tensor = pickle.loads(buffer.tobytes())
    copied_tensor = deepcopy(tensor)
    del buffer
    shm.close()
    return copied_tensor


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


def combine_micro_batches(micro_batches):
    """
    Combines the micro-batch outputs into a single output.
    """
    if isinstance(micro_batches[0], torch.Tensor):
        # If outputs are tensors, concatenate them along the batch dimension
        return torch.cat(micro_batches, dim=0)

    elif isinstance(micro_batches[0], ModelOutput):
        combined_output = defaultdict(list)

        for output in micro_batches:
            for key, value in output.items():
                combined_output[key].append(value)

        # Concatenate fields that are tensors
        final_output = {}
        for key, value in combined_output.items():
            if isinstance(value[0], torch.Tensor):
                # Handle zero-dimensional tensors
                if value[0].dim() == 0:
                    final_output[key] = torch.stack(value)
                else:
                    final_output[key] = torch.cat(value, dim=0)
            else:
                final_output[key] = value  # Leave as is if not a tensor

        return type(micro_batches[0])(**final_output)

    else:
        raise TypeError("Unsupported output type")


class DistributedModel(nn.Module):
    """
    DistributedModel:
        Is able to host offloaded submodules along with local operations. Can be spawned
        by a Worker or a User. Host is referred to as the 'master' node.
    """
    def __init__(
        self,
        node_requests,
        node_responses,
        model: nn.Module,
        batch_size: int,
        micro_batch_size: int,
        config=None,
        device=None
    ):
        super(DistributedModel, self).__init__()

        # Node process communication params
        self.node_requests = node_requests
        self.node_responses = node_responses

        assert (
            batch_size % micro_batch_size == 0
        ), "Micro-batch must be divisible by batch."

        self.model = model
        self.user_memory = get_gpu_memory()

        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size

        self.model.n_batch = 0
        self.model.forward_queues = {}
        self.model.backward_queues = {}
        self.model.intermediates = [
            []
        ]  # Queue to hold intermediates, must be converted into dictionary
        # of queues if we wish to perform multiple epochs concurrently

        self.distributed_graph = {}
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.spawn_workers = []

    def request_job(
        self,
        minibatch_size=1,
        microbatch_size=1,
        max_module_size=4e9,
        handle_layers=True,
    ):
        """Request job through smart contract and set up the relevant connections for a distributed model.
        Returns a distributed nn.Module with built-in RPC calls to workers."""
        # TODO Auto micro batch (data pipelines) selection based on job request (free = 2 max, paid is maximized to 16?)

        # Get model distribution schematic
        self.distributed_graph = self.parse_model(self.model, max_module_size, handle_layers=handle_layers)

        assert(
            not minibatch_size % microbatch_size,
            "Minibatch size must be divisible by microbatch size!"
        )

        n_pipelines = minibatch_size // microbatch_size

        # Get offloaded GPU usage for network
        capacity = (sum(v["size"] for v in self.distributed_graph.values()) * n_pipelines)

        job_info = (n_pipelines, capacity, self.distributed_graph)
        config = self.send_request("request_job", job_info)
        self.distribute_model(config)

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
        self.model.outputs = [None] * n_micro_batches
        self.model.intermediates = {m: [[]] for m in range(n_micro_batches)}

        # Spawn separate threads for each micro batch for pipeline parallelism
        threads = []
        for micro, batch in enumerate(micro_batches):
            # self.perform_micro_forward(micro, batch)
            t = threading.Thread(
                target=self._perform_micro_forward, args=(micro, batch), kwargs=kwargs
            )
            t.start()
            threads.append(t)
            time.sleep(0.5)

        for t in threads:
            t.join()

        return combine_micro_batches(self.model.outputs)

    def _perform_micro_forward(self, micro, *args, **kwargs):
        THREAD_STORAGE.micro = micro
        x = self.model(*args, **kwargs)
        self.model.outputs[micro] = x

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
                target=self._perform_micro_backward, args=(micro, loss[micro])
            )
            t.start()
            threads.append(t)
            time.sleep(0.5)

        for t in threads:
            t.join()

    def _perform_micro_backward(self, micro, loss):
        """
        Process each backwards stream
        """
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
                worker_id = self.distributed_graph[module_id]

                self.send_request("send_backward", (worker_id, loss, tag))

                if self.model.backward_relays.not_empty:
                    loss = self.model.backward_relays.get()

            # Len vals 2 means backwards pass of first & last submodule
            elif len(vals) == 2:
                val1, val2 = vals

                # Pass of the first submodule / section contains tensor in the first position
                if isinstance(val1, torch.Tensor):
                    assoc_output, _ = vals
                    loss = assoc_output.backward(loss, retain_graph=True)
                # Pass of the last section
                else:
                    module_id_bytes, assoc_input = vals
                    tag = [self.model.n_batch, micro, module_id_bytes]

                    # If there are remaining computations between output and last submodule
                    if loss.grad_fn is not None:
                        loss.backward()
                        loss = assoc_input.grad

                    start_time = time.time()
                    worker_id = self.distributed_graph[module_id_bytes]["workers"][micro]
                    key = tag[:2] + [module_id_bytes, module_id_bytes]

                    self.send_request("send_backward", (worker_id, (loss, None, tag)))

                    # Wait for response, change to appending waiting thread to list in master
                    waiting = True
                    while waiting:
                        time.sleep(0.5)
                        args = self.send_request("check_backward", key)

                        if args is not None:
                            waiting = False
                            shape, size, dtype, name = args
                            loss = get_from_shared_memory(size, name)

                        if time.time() - start_time >= MAX_WAIT_TIME:
                            # Logic here to request another worker take his place
                            waiting = False

                        time.sleep(0.1)

                    self.send_request("release_memory", tag)

            elif len(vals) == 1:
                assoc_input = vals[0]
                if isinstance(assoc_input, torch.Tensor):
                    assoc_input.backward(loss)
            else:
                raise "Expect vals to be of length 2 or 3."

    def get_info_from_module_id(self, mod_id: list, micro: int = None):
        for info in self.distributed_graph.values():
            if mod_id == info["mod_id"]:
                if micro:
                    return info["id_hash"], info["workers"][micro]
                else:
                    return info["id_hash"]

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
                    self.send_request("update_train", (children.worker_id, mode, children.module_id))
                    t = threading.Thread(
                        target=self._wait_for_state_update,
                        args=(children.module_id, mode),
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
                        self.send_request("request_parameters", (children.worker_node, children.module_id))
                        t = threading.Thread(
                            target=self._wait_for_parameters, args=(children.module_id,)
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

    def _wait_for_parameters(self, module_id: bytes):
        """
        Waits until parameters have been received from the specified offloaded module
        """
        # Wait for response, change to appending waiting thread to list in master
        start_time = time.time()
        waiting = True
        while waiting:
            time.sleep(0.25)
            args = self.send_request("check_parameters", module_id)

            if args is not None:
                waiting = False
                size, name = args
                output = get_from_shared_memory(size, name)

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

        self.send_request("release_memory", b"P" + module_id)

    def _wait_for_state_update(self, module_id, state):
        """
        Waits until a state update has been confirmed from the specified offloaded module
        """
        # Wait for response, change to appending waiting thread to list in master
        start_time = time.time()
        waiting = True
        while waiting:
            time.sleep(0.25)
            args = self.send_request("check_train", module_id)

            if args is not None:
                if args == state:
                    waiting = False

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

    def get_node_most_memory(self):
        # gets node with most memory, returns key and memory
        key = max(
            self.master_node.nodes, key=lambda x: self.master_node.nodes[x]["memory"]
        )
        return key, self.master_node.nodes[key]["memory"]

    def distribute_model(self, config=None):
        # Retrieve model names and assign workers to offload. Contact candidate workers
        # and ensure they are ready to receive the model / train

        if config is None:
            config = self.parse_model(self.model, self.max_module_size)

        self.distributed_graph = config

        for mod_info in config.values():
            mod_id = mod_info["mod_id"]
            worker_id = mod_info["workers"][0]
            self.wrap_module(mod_id, worker_id)

        return 0, 0

    def parse_model(
        self,
        model,
        max_module_size,
        config: dict = None,
        ids: list = None,
        handle_layers=True,
        handled_layer=False,
    ) -> dict:
        """
        Parse model based on some minimum submodule size and return a config file containing the
        distributed model configuration.
        # TODO update memory of semi-offloaded modules
        """

        if config is None:
            config = {}
        if ids is None:
            ids = []

        # Create offloaded module data structure for config file
        def create_offloaded(module: nn.Module, module_index: list, module_size: int):
            module_id = (
                hashlib.sha256(str(random.random()).encode()).hexdigest().encode()
            )
            data = {
                "type": "offloaded",
                "id_hash": module_id,
                "module": f"{type(module)}".split(".")[-1].split(">")[0][
                    :-1
                ],  # class name
                "mod_id": module_index,
                "size": module_size,
                "parameters": {},
                "workers": [],
                "training": True
            }
            return module_id, data

        # Create user-loaded module data structure for config file
        def create_loaded(module: nn.Module, module_index: list, module_size: int):
            module_id = (
                hashlib.sha256(str(random.random()).encode()).hexdigest().encode()
            )
            data = {
                "type": "loaded",
                "id_hash": module_id,
                "module": f"{type(module)}".split(".")[-1].split(">")[0][
                    :-1
                ],  # class name
                "mod_id": module_index,
                "size": module_size,
                "parameters": {},
                "workers": [],
            }
            return module_id, data

        named_children = list(model.named_children())
        model_size = estimate_memory(model)

        # If we do not want to handle initial layers and model can fit on worker
        if handle_layers is False and model_size <= max_module_size:
            k, v = create_offloaded(model, [-1], model_size)
            config[k] = v

        # Break first model into children
        for i in range(len(named_children)):
            # Unpack module info
            name, submodule = named_children[i]
            module_memory = estimate_memory(submodule)

            # Update current module id
            new_ids = ids + [i]

            module_type = f"{type(submodule)}".split(".")[-1].split(">")[0][:-1]

            # Try to handle on user if specified
            if handle_layers:
                # If user can handle the layer
                if self.user_memory >= module_memory:
                    self.user_memory -= module_memory
                    k, v = create_loaded(submodule, new_ids, module_memory)
                    config[k] = v
                    handled_layer = True
                    continue

                # Break down model further if we haven't handled first layer
                elif handled_layer is False:
                    sub_config = self.parse_model(
                        submodule,
                        max_module_size,
                        config,
                        new_ids,
                        True,
                        False,
                    )
                    k, v = create_loaded(submodule, new_ids, module_memory)
                    v["subconfig"] = sub_config
                    config[k] = v
                    continue

            # Append module id to offloaded config if meets the minimum size
            if module_memory <= max_module_size:
                k, v = create_offloaded(submodule, new_ids, module_memory)
                config[k] = v

            # Recursively break down model if too large
            else:
                sub_config = self.parse_model(
                    submodule, max_module_size, config, new_ids, True, True
                )
                k, v = create_loaded(submodule, new_ids, module_memory)
                v["subconfig"] = sub_config
                config[k] = v

        return config

    def wrap_module(self, module_id: list, worker_id):
        child_module, module_name = access_module(self.model, module_id)
        parent_module = (
            access_module(self.model, module_id[:-1])[0]
            if len(module_id) > 1
            else self.model
        )

        offloaded_module = OffloadedModule(self, worker_id)

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
        module_hash = self.get_info_from_module_id(module_id)
        offloaded_module.spawn_worker(child_module, module_hash)

        # spawn_worker = threading.Thread(target=offloaded_module.spawn_worker, args=(child_module, module_id))
        # spawn_worker.start()
        # self.spawn_workers.append(spawn_worker)

    def send_request(self, request_type, args):
        """
        Sends a request to the node and waits for the response.
        """
        request = {"type": request_type, "args": args}
        try:
            self.node_requests.put(request)
            response = self.node_responses.get()  # Blocking call, waits for response
            return response["return"]
        except Exception as e:
            print(f"Error sending request: {e}")
            return {"error": str(e)}


class OffloadedModule(nn.Module):
    """
    OffloadedModule:
        A module wrapper that handles offloading on the master node without disrupting normal
        pytorch model flow.
    """

    def __init__(self, parent_model: DistributedModel, worker_id):
        super(OffloadedModule, self).__init__()

        self.parent_model = parent_model
        self.worker_id = worker_id
        self.module_id = None

    def spawn_worker(self, module: nn.Module, module_id: bytes):
        # # Initialize a threading Timer to monitor the loading process
        # timer = threading.Timer(MAX_WAIT_TIME, self.handle_timeout)
        # timer.start()

        # try:
        # Send the module to the worker node
        self.module_id = module_id
        module.id = self.module_id

        self.parent_model.send_request("send_model", (module, self.worker_id))

        # # Module loaded successfully
        # except TimeoutError:
        #     # Timeout occurred, handle it
        #     self.handle_timeout()
        #
        # finally:
        #     # Stop the timer as loading completed (or failed)
        #     timer.cancel()

    def handle_timeout(self):
        # Timeout occurred, switch to another worker TODO
        new_worker_node = self.master_node.select_candidate_worker()
        if new_worker_node:
            self.worker_node = new_worker_node["connection"]
        else:
            # No available workers found, handle the situation accordingly
            pass

    def forward(self, *args, **kwargs):
        start_time = time.time()
        n_iter = self.parent_model.model.n_batch
        n_micro = getattr(THREAD_STORAGE, "micro", None)
        n_queued = self.parent_model.model.forward_queues[n_micro].qsize()

        tag = [n_iter, n_micro, self.module_id]

        # Store the intermediate tensor for backwards pass
        self.parent_model.model.intermediates[n_micro][-1].extend(
            [handle_output(args), self.module_id]
        )

        detached_args = handle_output(args).clone().detach()

        # Relay forward pass to next node
        self.parent_model.send_request("send_forward", (self.worker_id, (detached_args, kwargs, tag)))

        # Wait for response, change to appending waiting thread to list in master
        waiting = True
        while waiting:
            time.sleep(0.5)
            key = (n_iter, n_micro, self.module_id)
            args = self.parent_model.send_request("check_forward", key)

            if args is not None:
                waiting = False
                shape, size, dtype, name = args
                output = get_from_shared_memory(size, name)

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

            time.sleep(0.1)

        self.parent_model.send_request("release_memory", key)
        inter_storage = [
            self.module_id,
            handle_output(output),
        ]  # Store associated output

        # Store intermediates and connection for backwards pass
        self.parent_model.model.intermediates[n_micro].append(inter_storage)

        return output
