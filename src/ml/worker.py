from src.mpc.shared_memory import get_from_shared_memory, store_in_shared_memory
from src.ml.utils import *

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

        # torch._dynamo.config.suppress_errors = True
        # torch.set_num_threads(8)

    def train_loop(self):
        while not self.terminate:
            for module_id in list(self.modules.keys()):
                module = self.modules.get(module_id)

                if not module.backward_queue.empty():
                    n_micro_batch = module.n_micro_batch
                    next_node = module.host

                    if self.modules[module_id].training:
                        with self.lock:
                            tag, loss_relay = module.backward_queue.get()
                            loss = get_from_shared_memory(loss_relay[0], loss_relay[1])
                            microbatch = tag[1]

                            if isinstance(loss, tuple):
                                loss = tuple([l.to(self.device) for l in loss if isinstance(l, torch.Tensor)])
                            else:
                                loss = loss.to(self.device)

                            inter_tag = tuple(tag)
                            assoc_input, assoc_output = module.intermediates.pop(inter_tag)

                            assoc_input = assoc_input.to(self.device)
                            assoc_output = assoc_output.to(self.device)

                            assoc_output.backward(loss)

                            dvalues = detach_tensor(assoc_input.grad)

                            del assoc_input, assoc_output
                            torch.cuda.empty_cache()

                            size, name = store_in_shared_memory(dvalues)
                            self.send_request("send_backward", (next_node, size, name, tag))

                            if n_micro_batch - 1 == microbatch:
                                self.optimizers[module_id].step()

                            # Clear variables for backward pass
                            del loss, dvalues, tag, loss_relay, inter_tag, next_node
                            torch.cuda.empty_cache()

                if not module.forward_queue.empty():
                    with self.lock:
                        key, (size, name) = module.forward_queue.get()
                        tensor = get_from_shared_memory(size, name)

                        if isinstance(tensor, tuple):
                            args, kwargs = tensor
                        else:
                            args = tensor
                            kwargs = {}

                        inp = enable_grad(attach_tensor(args, self.device))
                        kwargs = enable_grad(attach_tensor(kwargs, self.device))

                        out = module(inp, **kwargs)

                        if self.modules[module_id].training:
                            module.intermediates[key] = [inp, handle_output(out).to(self.device)]

                        detached_out = detach_tensor(out)
                        size, name = store_in_shared_memory(detached_out)

                        self.send_request("send_forward", (module.host, size, name, key))

                        del inp, out, detached_out, args, kwargs, key, size, name  # Clear memory
                        torch.cuda.empty_cache()

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

                    module.forward_queue = queue.Queue()
                    module.backward_queue = queue.Queue()
                    module.intermediates = {}
                    module.host = node_id

                    with self.lock:
                        # self.optimizers[module_id] = torch.optim.Adam(module.parameters(), lr=2e-5)
                        self.modules[module_id] = module
                        self.send_request("module_loaded", module_id)

            if self.modules:
                for module_id in self.modules.keys():
                    module = self.modules[module_id]

                    args = self.send_request("check_train", module_id)

                    if isinstance(args, bool):
                        module.training = args

                    args2 = self.send_request("check_forward", module_id)
                    if args2:
                        module.forward_queue.put(args2)

                    args3 = self.send_request("check_backward", module_id)
                    if args3:
                        module.backward_queue.put(args3)

            counter += 1
            time.sleep(0.1)

    def run(self):
        node_check_thread = threading.Thread(target=self.check_node)
        node_check_thread.start()

        self.train_loop()

        node_check_thread.join()
