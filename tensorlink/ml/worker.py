from tensorlink.mpc.shared_memory import get_from_shared_memory, store_in_shared_memory
from tensorlink.ml.utils import *
import threading
import torch
import queue
import time
import os


class DistributedWorker:
    def __init__(self, node_requests, node_responses, mpc_lock):
        self.node_requests = node_requests
        self.node_responses = node_responses
        self.mpc_lock = mpc_lock

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
                    n_micro_batch = module.n_micro_batch
                    next_node = module.host

                    if self.modules[module_id].training:
                        # Critical section: lock the shared resources only when necessary
                        with self.lock:
                            tag, loss_relay = module.backward_queue.get()

                        # Load loss and move to the device
                        loss = get_from_shared_memory(loss_relay[0], loss_relay[1])
                        if isinstance(loss, tuple):
                            loss = tuple([l.to(self.device) for l in loss if isinstance(l, torch.Tensor)])
                        else:
                            loss = loss.to(self.device)

                        inter_tag = tuple(tag)
                        assoc_input, assoc_output = module.intermediates.pop(inter_tag)

                        assoc_input = assoc_input.to(self.device)
                        assoc_output = assoc_output.to(self.device)

                        # Backward pass with CUDA synchronization for accurate profiling
                        if self.device.type != "cpu":
                            torch.cuda.synchronize()
                        assoc_output.backward(loss[-1].view_as(assoc_output))

                        if self.device.type != "cpu":
                            torch.cuda.synchronize()

                        # Detach gradients and prepare for next node
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

    def check_node(self):
        update_check_interval = 20
        counter = 0

        while not self.terminate:
            if counter % update_check_interval == 0:
                args = self.send_request("check_module", None)

                if isinstance(args, tuple):
                    file_name, module_id, node_id = args
                    module = torch.load(file_name).to(self.device)
                    os.remove(file_name)

                    # Initialize module queues and states
                    module.forward_queue = queue.Queue()
                    module.backward_queue = queue.Queue()
                    module.intermediates = {}
                    module.host = node_id

                    with self.lock:
                        self.modules[module_id] = module
                        self.send_request("module_loaded", module_id)

            # Process training, forward, and backward queues
            if self.modules:
                for module_id in self.modules.keys():
                    module = self.modules[module_id]

                    # Check if module is in training mode
                    is_training = self.send_request("check_train", module_id)
                    if isinstance(is_training, bool):
                        module.training = is_training

                    # Handle forward queue
                    forward_task = self.send_request("check_forward", module_id)
                    if forward_task:
                        module.forward_queue.put(forward_task)

                    # Handle backward queue
                    backward_task = self.send_request("check_backward", module_id)
                    if backward_task:
                        module.backward_queue.put(backward_task)

            counter += 1
            time.sleep(0.1)

    def run(self):
        node_check_thread = threading.Thread(target=self.check_node)
        node_check_thread.start()

        self.train_loop()

        node_check_thread.join()
