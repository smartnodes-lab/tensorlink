import logging

from tensorlink.mpc.shared_memory import get_from_shared_memory, store_in_shared_memory
from tensorlink.ml.utils import *
from collections import deque
import threading
import pickle
import torch
import queue
import json
import time
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs("snapshots", exist_ok=True)


class DistributedWorker:
    def __init__(self, node_requests, node_responses, mpc_lock):
        self.node_requests = node_requests
        self.node_responses = node_responses
        self.mpc_lock = mpc_lock
        self.rolling_buffer = deque(maxlen=10)
        self.storage_path = "./snapshots"

        self.modules = {}
        self.optimizers = {}
        self.terminate = False
        self.lock = threading.Lock()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_loop(self):
        while not self.terminate:
            for module_id in list(self.modules.keys()):
                module = self.modules.get(module_id)

                # Handle backward pass
                if not module.backward_queue.empty():
                    n_batch = module.n_batch
                    next_node = module.host

                    if self.modules[module_id].training:
                        # Critical section: lock the shared resources only when necessary
                        with self.lock:
                            tag, loss_relay = module.backward_queue.get()

                            # Load loss and move to the device
                            loss = get_from_shared_memory(loss_relay[0], loss_relay[1])

                            if isinstance(loss, tuple):
                                loss = tuple([l.to(self.device) for l in loss if isinstance(l, torch.Tensor)])

                                if len(loss) == 1:
                                    loss = loss[-1]
                                #
                                # if loss.grad is None:

                            else:
                                loss = loss.to(self.device)

                            inter_tag = tuple(tag)
                            assoc_input, assoc_output = module.intermediates.pop(inter_tag)

                            assoc_input = assoc_input.to(self.device)
                            assoc_output = assoc_output.to(self.device)

                            # Backward pass with CUDA synchronization for accurate profiling
                            if self.device.type != "cpu":
                                torch.cuda.synchronize()

                            assoc_output.backward(loss)

                            if self.device.type != "cpu":
                                torch.cuda.synchronize()

                            # Detach gradients and prepare for next node
                            if assoc_input.grad is None:
                                dvalues = detach_tensor(torch.zeros_like(assoc_input, dtype=torch.float32))
                            else:
                                dvalues = detach_tensor(assoc_input.grad)

                            # Clean up to avoid memory leaks
                            del assoc_input, assoc_output

                            # Store gradients in shared memory and send to next node
                            size, name = store_in_shared_memory(dvalues)
                            self.send_request("send_backward", (next_node, size, name, tag))

                            # Clear memory, but avoid excessive cache clearing
                            if self.device.type != "cpu":
                                torch.cuda.empty_cache()

                # Handle forward pass
                if not module.forward_queue.empty():
                    with self.lock:
                        key, (size, name) = module.forward_queue.get()

                        tensor = get_from_shared_memory(size, name)
                        if isinstance(tensor, tuple):
                            args, kwargs = tensor
                        else:
                            args = tensor
                            kwargs = {}

                        # Enable gradient tracking and move to device
                        inp = enable_grad(attach_tensor(args, self.device))
                        kwargs = enable_grad(attach_tensor(kwargs, self.device))

                        # Forward pass with CUDA synchronization for accurate profiling
                        if self.device.type != "cpu":
                            torch.cuda.synchronize()

                        out = module(inp, **kwargs)

                        if self.device.type != "cpu":
                            torch.cuda.synchronize()

                        # Store intermediate results if training
                        if self.modules[module_id].training:
                            module.intermediates[key] = [inp, handle_output(out).to(self.device)]

                        # Detach and store output
                        detached_out = detach_tensor(out)
                        size, name = store_in_shared_memory(detached_out)
                        self.send_request("send_forward", (module.host, size, name, key))

                        # Clean memory efficiently
                        if self.device.type != "cpu":
                            torch.cuda.empty_cache()

                        # self.store_snapshot(module_id, inp, out, key[0], key[1])

                        if module.training:
                            module.n_batch += 1

    def send_request(self, request_type, args):
        request = {"type": request_type, "args": args}
        try:
            self.mpc_lock.acquire()
            self.node_requests.put(request)
            response = self.node_responses.get()  # Blocking call, waits for response
        except Exception as e:
            print(f"Error sending request: {e}")
            response = {"response": str(e)}
        finally:
            self.mpc_lock.release()

        return response["return"]

    def store_snapshot(self, module_id, _input, _output, epoch, micro):
        # Ensure the snapshots directory exists
        os.makedirs("snapshots", exist_ok=True)

        # Get parameters (state_dict) and convert tensors to a serializable format
        params = {k: v.cpu().numpy().tolist() for k, v in self.modules[module_id].state_dict().items()}

        # Prepare snapshot data
        snapshot = {
            "id": module_id,
            "params": params,
            "input": _input.cpu().numpy().tolist(),  # Assuming _input is a tensor
            "output": _output.cpu().numpy().tolist(),  # Assuming _output is a tensor
            "epoch": epoch,
            "micro": micro
        }

        # Define the filename
        file_path = os.path.join("snapshots", f"{module_id}_epoch{epoch}_micro{micro}.json")

        # Write the snapshot to a JSON file
        try:
            with open(file_path, "w") as f:
                json.dump(snapshot, f)
            print(f"Snapshot saved successfully: {file_path}")
        except IOError as e:
            print(f"Error saving snapshot: {e}")

    def load_module(self, file_name, module_id, node_id):

        # Load the module in a separate thread
        module = torch.load(file_name).to(self.device)
        os.remove(file_name)

        # Initialize queues and states
        module.forward_queue = queue.Queue()
        module.backward_queue = queue.Queue()
        module.intermediates = {}
        module.host = node_id

        self.modules[module_id] = module
        self.optimizers[module_id] = module.optimizer
        delattr(module, "optimizer")
        self.send_request("module_loaded", module_id)

    def check_node(self):
        update_check_interval = 50
        counter = 0

        while not self.terminate:
            if counter % update_check_interval == 0:
                args = self.send_request("check_module", None)

                if isinstance(args, tuple):
                    file_name, module_id, node_id = args
                    self.load_module(file_name, module_id, node_id)

                # Check for job completion/deletion requests
                elif isinstance(args, str):
                    if args in self.modules:
                        del self.modules[args]
                        del self.optimizers[args]
                        self.send_request("debug_print", (f"Module {args} removed.",))

                # Check for node termination requests
                self.check_for_termination()

            # Process training, forward, and backward queues
            if self.modules:
                for module_id in self.modules.keys():
                    module = self.modules[module_id]

                    # Check if module is in training mode
                    is_training = self.send_request("check_train", module_id)
                    if isinstance(is_training, bool):
                        module.training = is_training

                    # Check for parameters requests
                    params_req = self.send_request("check_parameters_request", module_id)
                    if params_req:
                        self.send_request("debug_print", ("DistributedWorker -> Sending parameters.",))
                        with open(f"parameters_{module_id}", "wb") as file:
                            torch.save(module.state_dict(), file)

                        self.send_request("send_parameters", (module.host, module_id))

                    # Handle forward queue
                    forward_task = self.send_request("check_forward", module_id)
                    if forward_task:
                        module.forward_queue.put(forward_task)

                    # Handle backward queue
                    backward_task = self.send_request("check_backward", module_id)
                    if backward_task:
                        module.backward_queue.put(backward_task)

                    state_update = self.send_request("check_state_update", module_id)
                    if state_update:
                        with self.lock:
                            if state_update[0] == "init":
                                optimizer_kwargs = state_update[1]
                                self.optimizers[module_id] = self.optimizers[module_id](module.parameters(), **optimizer_kwargs)
                                self.send_request("debug_print",
                                                  ("DistributedWorker -> Initialized optimizer.", "bright_blue",
                                                   logging.INFO))
                                self.send_request("optimizer_response", (module_id, "loaded"))

                            elif state_update[0] == "step":
                                closure = state_update[1]
                                self.optimizers[module_id].step(closure)
                                self.send_request("debug_print",
                                                  ("DistributedWorker -> Optimizer stepped.", "bright_blue",
                                                   logging.INFO))
                                self.send_request("optimizer_response", (module_id, "stepped"))

                            elif state_update[0] == "zero_grad":
                                self.optimizers[module_id].zero_grad()
                                self.send_request("debug_print",
                                                  ("DistributedWorker -> Optimizer zeroed.", "bright_blue",
                                                   logging.INFO))
                                self.send_request("optimizer_response", (module_id, "zeroed"))

            counter += 1
            time.sleep(0.05)

    def check_for_termination(self):
        # Send a request to check if the node is shutting down
        shutdown_signal = self.send_request("check_shutdown", None)
        if shutdown_signal:  # Assuming shutdown_signal is True or some indication of shutdown
            print("Termination signal received. Shutting down DistributedWorker process...")
            self.terminate = True

    def run(self):
        node_check_thread = threading.Thread(target=self.check_node)
        node_check_thread.start()

        self.train_loop()

        node_check_thread.join()
