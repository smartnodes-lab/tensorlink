from tensorlink.ml.optim import create_distributed_optimizer, DistributedParameter
from tensorlink.ml.utils import *
from tensorlink.mpc.shared_memory import get_from_shared_memory, store_in_shared_memory

from transformers.utils import ModelOutput
from transformers import PreTrainedModel
from collections import defaultdict
from typing import Generator, OrderedDict
import torch.nn as nn
import threading
import hashlib
import random
import torch
import queue
import time
import json
import os
import gc


MAX_WAIT_TIME = 300


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


class CustomAutogradRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model, output_tensor):
        """
        Forward pass, save the model in the context (ctx) for use during the backward pass.
        """
        ctx.model = model
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass, call the model's custom backward method.
        """
        # Call the model's custom backward method
        grad_input = ctx.model.backward(grad_output)

        # Return gradients for model and output_tensor
        return None, grad_input


class DistributedModel(nn.Module):
    """
    DistributedModel:
        Is able to host offloaded submodules along with local operations. Can be spawned
        by a Worker or a User. Host is referred to as the 'master' roles.
    """
    def __init__(
        self,
        node_requests,
        node_responses,
        mpc_lock,
        model: nn.Module,
        n_pipelines: int,
        device=None
    ):
        super(DistributedModel, self).__init__()
        self.name = str(model).split("(")[0]

        # Node process communication params
        self.node_requests = node_requests
        self.node_responses = node_responses
        self.mpc_lock = mpc_lock

        self.model = model
        self.user_memory = get_gpu_memory()

        self.worker_info = {}

        self.n_pipelines = n_pipelines
        self.n_datalines = 1

        self.distributed_graph = {}
        self.parameters_storage = {}

        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.optimizer = None

    def forward(self, *args, **kwargs):
        """
        Performs the forward pass through the model.
        - Splits input into micro-batches and runs them in parallel.
        - Creates multiple parallel streams of workers for model parallel acceleration
        """
        if isinstance(args, tuple) and len(args) == 1:
            args = args[0]

        batch_size = get_batch_size(args)

        assert (
            batch_size % self.n_datalines == 0
        ), "Batch size must be divisible by n_datalines!"

        assert (
            batch_size // self.n_datalines % self.n_pipelines == 0
        ), "Minibatch size must be divisible by n_pipelines!"

        # Split the input tensor and kwargs into micro-batches
        micro_batch_args = chunk(args, self.n_pipelines)
        micro_batch_kwargs = chunk(kwargs, self.n_pipelines)

        # Create queues for forward and backward passes
        self.model.forward_queues = {m: queue.Queue() for m in range(self.n_pipelines)}
        self.model.backward_queues = {m: queue.Queue() for m in range(self.n_pipelines)}
        self.model.outputs = [None] * self.n_pipelines
        self.model.intermediates = {m: [] for m in range(self.n_pipelines)}

        # Spawn separate threads for each micro batch for pipeline parallelism
        threads = []
        for micro in range(self.n_pipelines):
            micro_args = micro_batch_args[micro]
            micro_kwargs = micro_batch_kwargs[micro]
            t = threading.Thread(
                target=self._perform_micro_forward, args=(micro, micro_args), kwargs=micro_kwargs
            )
            t.start()
            threads.append(t)
            time.sleep(0.1)

        for t in threads:
            t.join()

        if not self.training:
            self.model.n_batch += 1

        combined_output = combine_micro_batches(self.model.outputs)
        output_tensor = handle_output(combined_output)
        custom_grad_output = CustomAutogradRouter.apply(self, output_tensor)
        return replace_output_with_custom_grad(combined_output, custom_grad_output)

    def _perform_micro_forward(self, micro, *args, **kwargs):
        kwargs["micro"] = micro
        x = self.model(*args, **kwargs)
        self.model.outputs[micro] = x

    def backward(self, loss):
        """
        Performs the backward pass through the model.
        - Splits input into micro-batches and runs them in parallel.
        """
        if hasattr(loss, "micro_loss"):
            micro_losses = loss.micro_loss
        else:
            # Split the combined loss into micro-batch-specific losses
            micro_losses = split_into_micro_batches(loss, self.n_pipelines)

        threads = []
        for micro in range(self.n_pipelines):
            t = threading.Thread(
                target=self._perform_micro_backward, args=(micro, micro_losses[micro])
            )
            t.start()
            threads.append(t)
            time.sleep(0.1)

        for t in threads:
            t.join()

        self.model.n_batch += 1

    def _perform_micro_backward(self, micro, loss):
        """
        Process each backwards stream
        """

        # Get the oldest epoch intermediates from storage via dict key lookup (todo)
        while len(self.model.intermediates[micro]) > 0:
            vals = self.model.intermediates[micro].pop(-1)

            # Len vals 2 means backwards pass of first & last submodule
            if len(vals) == 2:
                val1, val2 = vals

                # Pass of the first submodule / section contains tensor in the first position
                if isinstance(val1, torch.Tensor):
                    assoc_output, _ = vals
                    loss = assoc_output.backward(loss, retain_graph=True)

                    if loss is None:
                        # If loss is None, use the associated output for backward pass
                        loss = assoc_output

                # Pass of the last section
                else:
                    module_id_bytes, assoc_input = vals
                    tag = [self.model.n_batch, micro, module_id_bytes]

                    # If there are remaining computations between output and last submodule
                    if loss.grad_fn is not None:
                        loss.backward()
                        loss = assoc_input.grad

                    start_time = time.time()
                    worker_id = self.distributed_graph[module_id_bytes]["workers"][-1]
                    key = tag[:2] + [module_id_bytes, module_id_bytes]

                    loss_bytes = json.dumps(tensor_to_bytes(loss)).encode()
                    size, shm_name = store_in_shared_memory(
                        loss_bytes,
                        encoded=True
                    )
                    self.send_request("send_backward", (worker_id, size, shm_name, tag))

                    # Wait for response, change to appending waiting thread to list in master
                    waiting = True
                    while waiting:
                        time.sleep(0.1)
                        args = self.send_request("check_backward", key)

                        if args is not None:
                            waiting = False
                            size, name = args
                            loss_bytes = get_from_shared_memory(size, name, encoded=True)
                            loss = bytes_to_tensor(loss_bytes)

                        if time.time() - start_time >= MAX_WAIT_TIME:
                            # Logic here to request another worker take his place
                            waiting = False

                    self.send_request("release_memory", ("backward_queue", module_id_bytes, tuple(tag)))

            elif len(vals) == 1:
                assoc_input = vals[0]
                if isinstance(assoc_input, torch.Tensor):
                    assoc_input.backward(loss)
            else:
                raise "Expect vals to be of length 1 or 2."

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
            if isinstance(module, OffloadedModule):
                self.send_request("update_train", (module.worker_id, mode, module.module_id))
                t = threading.Thread(
                    target=self._wait_for_state_update,
                    args=(module.module_id, mode),
                )
                t.start()
                threads.append(t)

            elif contains_offloaded(module):
                for child in module.children():
                    recurse_modules(child)

        recurse_modules(self.model)

        for thread in threads:
            thread.join()

    def eval(self):
        self.train(False)

    def children(self):
        # If the model is an instance of OffloadedModule, return an iterator with only itself.
        if isinstance(self.model, OffloadedModule):
            yield self.model  # Just yield the OffloadedModule, don't dive into its children.
        else:
            # Otherwise, yield the model itself and then recursively yield its children.
            yield self.model
            yield from self.model.children()

    def parameters(self, recurse: bool = True, distributed: bool = True, load: bool = True) -> Generator:
        """
        Collects parameters from all modules (including offloaded) asynchronously and returns them as a generator.
        """
        if distributed:
            parameters = []
            request_threads = []
            file_references = {}

            def collect_parameters(module):
                for child in module.children():
                    if isinstance(child, OffloadedModule):
                        # Asynchronously request parameters from offloaded modules
                        self.send_request("request_parameters", (child.worker_id, child.module_id))
                        # Start a thread to wait for each response
                        t = threading.Thread(target=self._wait_for_parameters, args=(child.module_id,))
                        t.start()
                        request_threads.append(t)
                    elif contains_offloaded(child):
                        collect_parameters(child)
                    else:
                        parameters.extend(list(child.parameters()))

            # Recursively collect parameters from main and offloaded modules
            collect_parameters(self)

            # Ensure all parameter requests are completed
            for thread in request_threads:
                thread.join()

            # Handle each module's parameters, loading only if `load` is True
            for module_id, module_info in self.distributed_graph.items():
                if module_info["type"] == "offloaded":
                    file_name = self.parameters_storage[module_id]
                    if load:
                        # Load parameters from file
                        with open(file_name, "rb") as f:
                            state_dict = torch.load(f)
                            parameters.extend(state_dict.values())
                    else:
                        # Store the reference to the file for lazy access
                        file_references[module_id] = file_name
                        os.makedirs("models", exist_ok=True)
                        os.makedirs(f"models/{self.name}", exist_ok=True)
                        os.rename(file_name, os.path.join("models", self.name, file_name.split("/")[1]))
                else:
                    # Directly collect parameters for in-memory modules
                    module, _ = access_module(self.model, module_info["mod_id"])
                    parameters.extend(list(module.parameters()))

            # Define a generator to yield either parameters or file references
            def parameter_generator():
                if load:
                    # Yield loaded or in-memory parameters directly
                    for param in parameters:
                        yield param
                else:
                    # Yield file references instead of loading parameters
                    for _module_id, file_name in file_references.items():
                        yield file_name  # Yield file paths if `load` is False

            return parameter_generator()

        # If not distributed, return parameters from the model directly
        return (param for param in self.model.parameters(recurse))

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
                file_name = args

                self.parameters_storage[module_id] = file_name

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

    def _wait_for_state_update(self, module_id, state):
        """
        Waits until a state update has been confirmed from the specified offloaded module
        """
        # Wait for response, change to appending waiting thread to list in master
        start_time = time.time()
        waiting = True
        while waiting:
            time.sleep(0.1)
            args = self.send_request("check_train", module_id)

            if args is not None:
                if args == state:
                    waiting = False

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

    def get_node_most_memory(self):
        # gets roles with most mpc, returns key and mpc
        key = max(
            self.master_node.nodes, key=lambda x: self.master_node.nodes[x]["mpc"]
        )
        return key, self.master_node.nodes[key]["mpc"]

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

        if len(config) == 1:
            module, module_name = access_module(self.model, [-1])
            setattr(module, "entire_model", True)

        self.model.n_batch = 0
        self.model.forward_queues = {}
        self.model.backward_queues = {}
        self.model.intermediates = [
            []
        ]  # Queue to hold intermediates, must be converted into dictionary
        # of queues if we wish to perform multiple epochs concurrently

        return 0, 0

    def parse_model(
        self,
        model: nn.Module,
        config: dict = None,
        ids: list = None,
        handle_layer: bool = True,
    ) -> dict:
        """
        Parse model based on some minimum submodule size and return a config file containing the
        distributed model configuration.
        # TODO Commented out model distribution schemes for data processing on user end.
        """
        if config is None:
            config = {}
            ids = []

        # Create offloaded module data structure for config file
        def create_offloaded(module: nn.Module, module_index: list, module_size: int):
            module_id = (
                hashlib.sha256(str(random.random()).encode()).hexdigest()
            )
            data = {
                "type": "offloaded",
                "id_hash": module_id,
                "module": f"{type(module)}".split(".")[-1].split(">")[0][
                          :-1
                          ],  # class name
                "mod_id": module_index,
                "size": module_size,
                "workers": [],
                "training": True
            }
            return module_id, data

        # Create user-loaded module data structure for config file
        def create_loaded(module: nn.Module, module_index: list, module_size: int):
            module_id = (
                hashlib.sha256(str(random.random()).encode()).hexdigest()
            )
            data = {
                "type": "loaded",
                "id_hash": module_id,
                "module": f"{type(module)}".split(".")[-1].split(">")[0][
                          :-1
                          ],  # class name
                "mod_id": module_index,
                "size": module_size,
                "workers": [],
            }
            return module_id, data

        # named_children = list(model.named_children())
        model_size = estimate_memory(model, self.training, batch_size=1024)
        assert model_size < 24e9, "Models must currently require under ~24Gb of GPU memory due to network availability."
        # max_worker = max(self.worker_info.items(), key=lambda x: x[1]["memory"])
        # max_worker_mem = max_worker[1]["memory"]

        # If we do not want to handle initial layers and model can fit on worker.
        # Temporarily, this is the only option possible as model size must be less than 24
        if model_size <= 24e9:
            k, v = create_offloaded(model, [-1], model_size)
            v["name"] = None

            if not isinstance(model, PreTrainedModel):
                try:
                    torch.jit.script(model)

                except Exception as e:
                    raise RuntimeError(
                        "The model cannot be processed for distributed training due to compatibility or security "
                        "constraints. Support for additional model types and improved security handling will be "
                        "included in future updates."
                    )

            else:
                v["name"] = model.name_or_path

            config[k] = v

        # Otherwise we break the module down into its components and handle TODO
        # else:
        #     for i in range(len(named_children)):
        #         # Unpack module info
        #         name, submodule = named_children[i]
        #         module_memory = estimate_memory(submodule)
        #
        #         # Update current module id
        #         new_ids = ids + [i]
        #         module_type = f"{type(submodule)}".split(".")[-1].split(">")[0][:-1]
        #
        #         # Try handling the layer with user memory if possible
        #         # if self.user_memory >= module_memory and handle_layer:
        #         #     self.user_memory -= module_memory
        #         #     k, v = create_loaded(submodule, new_ids, module_memory)
        #         #     config[k] = v
        #         #     handle_layer = False
        #
        #         if module_memory <= 24e9:
        #             k, v = create_offloaded(submodule, new_ids, module_memory)
        #             config[k] = v
        #         else:
        #             # Recursively break down larger modules that can't fit in memory
        #             sub_config = self.parse_model(submodule, config=config.copy(), ids=new_ids,
        #                                           handle_layer=handle_layer)
        #             k, v = create_loaded(submodule, new_ids, module_memory)
        #             v["subconfig"] = sub_config
        #             config[k] = v

        return config

    def wrap_module(self, module_id: list, worker_id):
        # Access the module and parent
        if module_id == [-1]:  # Handling the case of the full model
            child_module = self.model
            parent_module = None  # No parent when the full model is offloaded
        else:
            child_module, module_name = access_module(self.model, module_id)
            parent_module = (
                access_module(self.model, module_id[:-1])[0]
                if len(module_id) > 1
                else self.model
            )

        module_hash = self.get_info_from_module_id(module_id)

        # Assign metadata to child_module
        child_module.id = module_hash
        child_module.n_batch = 0

        file_name = f"{module_hash}_{worker_id}.pt"
        # TODO Custom pickle function OR send state dict and recreate a model on other side
        if isinstance(child_module, PreTrainedModel):
            state_dict = child_module.state_dict()
            state_dict["module_config"] = child_module.config.to_dict()
            state_dict["module_class"] = child_module.__class__.__name__
            torch.save(state_dict, file_name)  # Save the module to disk
        else:
            try:
                scripted_module = torch.jit.script(child_module)
                scripted_module.save(file_name)
            except Exception as e:
                raise RuntimeError(
                    "The model cannot be processed for distributed training due to compatibility or security "
                    "constraints. Support for additional model types and improved security handling will be included "
                    "in future updates."
                )

        module_info = str(child_module)
        offloaded_module = OffloadedModule(self, module_info, worker_id, module_hash)

        # Detach and clear the parameters of the child_module to free memory
        # for name, param in child_module.named_parameters():
        #     # Create the DistributedParameter.
        #     module_hash = hash(child_module)
        #     distributed_param = DistributedParameter(child_module, module_hash, worker_id, name)
        #
        #     # Use a safe name for attribute registration.
        #     safe_name = name.replace('.', '_')
        #     setattr(child_module, safe_name, distributed_param)
        #
        #     # Replace the parameter in _parameters.
        #     child_module._parameters[safe_name] = distributed_param
        #     offloaded_module.add_distributed_parameter(safe_name, distributed_param)
        #
        #     # Remove the original parameter using its original name.
        #     if name in child_module._parameters:
        #         del child_module._parameters[name]

        for param in child_module.parameters():
            param.detach_()  # Detach from the computation graph
            param.requires_grad = False  # Ensure no gradients are computed
            param.data = torch.empty(0)  # Clear the data in the parameter

        # Optional: Clear any buffers (like batch norm running stats) in the child module
        for buffer_name, buffer in child_module.named_buffers(recurse=False):
            setattr(child_module, buffer_name, torch.empty(0))

        # Clear the module's parameters explicitly
        del child_module
        gc.collect()  # Force garbage collection

        # Spawn a worker thread for the offloaded module
        offloaded_module.spawn_worker(file_name)

        if parent_module is not None:
            # Update the parent module's child module with the offloaded module
            if isinstance(parent_module, nn.ModuleList):
                parent_module[module_id[-1]] = offloaded_module
            else:
                child_name = list(parent_module.named_children())[module_id[-1]][0]
                setattr(parent_module, child_name, offloaded_module)
        else:
            setattr(self, "model", offloaded_module)

    def send_request(self, request_type, args):
        """
        Sends a request to the roles and waits for the response.
        """
        request = {"type": request_type, "args": args}
        try:
            self.mpc_lock.acquire()
            self.node_requests.put(request)
            response = self.node_responses.get()  # Blocking call, waits for response

        except Exception as e:
            print(f"Error sending request: {e}")
            response = {"error": str(e)}

        finally:
            self.mpc_lock.release()

        return response["return"]


class OffloadedModule(nn.Module):
    """
    OffloadedModule:
        A module wrapper that handles offloading on the master roles without disrupting normal
        pytorch model flow.
    """

    def __init__(self, parent_model: DistributedModel, module_name, worker_id, module_id: bytes):
        super(OffloadedModule, self).__init__()

        self.entire_model = False
        self.module_name = module_name.split("(")[0]
        self.module_info = module_name.split("(")[1][:-1]
        self.parent_model = parent_model
        self.worker_id = worker_id
        self.module_id = module_id
        self.n_batch = 0

    def children(self):
        # Print the offloaded module and the original model name
        # print(f"OffloadedModule: {self.module_name}")
        # Return an empty iterator to hide deeper children
        return iter([])

    def spawn_worker(self, name):
        # # Initialize a threading Timer to monitor the loading process
        # timer = threading.Tier(MAX_WAIT_TIME, self.handle_timeout)
        # timer.start()

        # try:
        # Send the module to the worker roles

        self.parent_model.send_request("send_model", (name, self.worker_id, self.module_id))

        # Wait for the module to be loaded on worker
        waiting = True
        start_time = time.time()
        while waiting:
            time.sleep(0.5)
            args = self.parent_model.send_request("check_loaded", (self.worker_id, self.module_id))

            if args is True:
                waiting = False

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

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
        n_batch = self.parent_model.model.n_batch
        n_micro = kwargs.pop("micro")
        n_queued = self.parent_model.model.forward_queues[n_micro].qsize()

        tag = [n_batch, n_micro, self.module_id]

        # Store the intermediate tensor for backwards pass
        if not self.entire_model:
            self.parent_model.model.intermediates[n_micro].append(
                [handle_output(args), self.module_id]
            )

        detached_args = handle_output(args).clone().detach()
        args_bytes = json.dumps(tensor_to_bytes(detached_args)).encode()
        kwargs_bytes = json.dumps(tensor_to_bytes(kwargs)).encode()
        forward_bytes = args_bytes + b"|" + kwargs_bytes

        size, shm_name = store_in_shared_memory(
            forward_bytes,
            encoded=True
        )

        # Relay forward pass to next roles
        self.parent_model.send_request("send_forward", (self.worker_id, size, shm_name, tag))

        # Wait for response, change to appending waiting thread to list in master
        waiting = True
        while waiting:
            time.sleep(0.1)
            key = (n_batch, n_micro, self.module_id)
            args = self.parent_model.send_request("check_forward", key)

            if args is not None:
                waiting = False
                size, name = args
                output_bytes = get_from_shared_memory(size, name)

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

        output = enable_grad(bytes_to_tensor(output_bytes))

        self.parent_model.send_request("release_memory", ("forward_queue", self.module_id, key))

        inter_storage = [
            self.module_id,
            handle_output(output),
        ]  # Store associated output

        # Store intermediates and connection for backwards pass
        self.parent_model.model.intermediates[n_micro].append(inter_storage)

        return output

    def add_distributed_parameter(self, name, distributed_param):
        """Register a DistributedParameter to the offloaded module."""
        self._parameters[name] = distributed_param

    def parameters(self, recurse=True):
        """Override parameters to return wrapped DistributedParameters."""
        return iter(self._parameters.values())

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Returns an empty dictionary for the OffloadedModule.
        This effectively hides the OffloadedModule from the model's state_dict.
        """
        return {}

    def __repr__(self):
        # Custom representation to prevent recursion when printing or debugging.
        return f"OffloadedModule(name={self.module_name}, worker_id={self.worker_id})"
