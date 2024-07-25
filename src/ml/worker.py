import torch
import queue
import time

from src.ml.model_analyzer import *
from src.mpc.shared_memory import get_from_shared_memory, store_in_shared_memory

import threading


class DistributedWorker:
    def __init__(self, node_requests, node_responses):
        self.node_requests = node_requests
        self.node_responses = node_responses

        self.modules = {}
        self.optimizers = {}
        self.terminate = False
        self.lock = threading.Lock()

    def train_loop(self):
        while not self.terminate:
            # Complete outstanding forward and backward passes
            for module_id, module in self.modules.items():
                # Complete any outstanding back propagations
                if module.backward_queue.empty() is False:

                    next_node = module.host
                    # self.optimizers[module_id].zero_grad()

                    # Grab backwards pass from forward node and our associated input/output from forward pass
                    tag, loss_relay = module.backward_queue.get()
                    loss = get_from_shared_memory(loss_relay[0], loss_relay[1])
                    inter_tag = tuple(tag)
                    assoc_input, assoc_output = module.intermediates[inter_tag]

                    # Continue backwards pass on our section of model
                    assoc_output.backward(
                        loss, retain_graph=True
                    )  # Do we need retain graph?
                    dvalues = assoc_input.grad

                    # Pass along backwards pass to next node
                    size, name = store_in_shared_memory(dvalues)
                    self.send_request("send_backward", (next_node, size, name, tag))

                    # self.optimizers[module_id].step()

                if module.forward_queue.empty() is False:

                    key, (size, name) = module.forward_queue.get()
                    tensor = get_from_shared_memory(size, name)

                    # Unpack queued forward pass unpack values (eg. mask, stride...)
                    if isinstance(tensor, tuple):
                        args, kwargs = tensor
                    else:
                        args = tensor
                        kwargs = {}

                    # Clear tensor of any previous info, set up for custom backward pass
                    inp = (
                        handle_output(args).clone().detach().requires_grad_()
                    )  # This should be done on sending node, not receiving
                    out = module(inp, **kwargs)

                    # Store output and input tensor for backward pass
                    module.intermediates[key] = [inp, handle_output(out)]
                    detached_out = detach_tensor(out)
                    size, name = store_in_shared_memory(detached_out)

                    # Relay forward pass to the next node
                    with self.lock:
                        self.send_request("send_forward", (module.host, size, name, key))

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

    def check_node(self):
        module_check_interval = 25
        counter = 0

        while not self.terminate:
            if self.modules == {} and counter % module_check_interval == 0:
                args = self.send_request("check_module", None)

                if isinstance(args, tuple):
                    size, name, module_id, node_id = args

                    module = get_from_shared_memory(size, name)
                    module.forward_queue = queue.Queue()
                    module.backward_queue = queue.Queue()
                    module.intermediates = {}
                    module.host = node_id

                    # self.optimizers[module_id] = torch.optim.Adam(module.parameters(), lr=2e-5)

                    self.modules[module_id] = module
                    self.send_request("module_loaded", module_id)

            if self.modules:
                for module_id in self.modules.keys():
                    args = self.send_request("check_forward", module_id)
                    if args:
                        module.forward_queue.put(args)

                    args2 = self.send_request("check_backward", module_id)
                    if args2:
                        module.backward_queue.put(args2)

            counter += 1
            time.sleep(0.1)

    def run(self):
        node_check_thread = threading.Thread(target=self.check_node)

        node_check_thread.start()

        self.train_loop()

        node_check_thread.join()
