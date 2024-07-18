from src.ml.model_analyzer import get_gpu_memory, handle_output
from src.p2p.smart_node import SmartNode
from src.p2p.connection import Connection

from multiprocessing import shared_memory, Lock
import torch.optim as optim
import torch.nn as nn
import threading
import pickle
import queue
import torch
import time


def format_size(size_bytes):
    """
    Format the size to display in GB, MB, or KB with one decimal place.
    """
    if size_bytes >= 1e9:
        return f"{round(size_bytes / 1e9, 1)} GB"
    elif size_bytes >= 1e6:
        return f"{round(size_bytes / 1e6, 1)} MB"
    elif size_bytes >= 1e3:
        return f"{round(size_bytes / 1e3, 1)} KB"
    else:
        return f"{size_bytes} bytes"


class TorchNode(SmartNode):
    def __init__(
            self,
            request_queue,
            response_queue,
            debug: bool = False,
            max_connections: int = 0,
            upnp=True,
            off_chain_test=False,
    ):
        super(TorchNode, self).__init__(
            debug=debug,
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
        )

        # Available GPU memory estimation
        self.available_memory = get_gpu_memory()

        self.request_queue = request_queue
        self.response_queue = response_queue
        self.memory_manager = {}

        # Pointers to model parameters in DistributedModels
        self.modules = {}
        self.optimizers = {}
        self.parameters = {}
        self.state_updates = {}

        # Master flag for handling different types of storage as master
        self.master = False

    def handle_data(self, data: bytes, node: Connection):
        try:
            handled = super().handle_data(data, node)
            ghost = 0

            if not handled:
                if b"LOADED" == data[:6]:
                    pickled = data[6:]
                    self.debug_print(
                        f"Successfully offloaded submodule to: {node.node_id}"
                    )
                    module_id = data[6:]
                    self.remove_request(node.node_id, b"MODULE")

                elif b"FORWARD" == data[:7]:
                    # Received a forward pass
                    size = data.__sizeof__() - 5
                    formatted_size = format_size(size)
                    self.debug_print(f"RECEIVED FORWARD: {formatted_size}")

                    if self.role == b"U":
                        # TODO we must check that the forward received corresponds to a sent pass/specific module
                        [n_iter, n_micro, module_id], tensor = pickle.loads(data[7:])
                        key = (n_iter, n_micro, module_id)

                        # Create shared memory block and store tensor
                        self.store_tensor_in_shared_memory(key, tensor)

                    # TODO we must check that the forward received corresponds to a sent pass/specific module
                    elif self.modules:
                        (n_iter, n_micro, module_id), tensor = pickle.loads(data[7:])
                        self.modules[module_id].forward_queues.put(
                            ([n_iter, n_micro], tensor)
                        )

                elif b"BACKWARD" == data[:8]:
                    # TODO same with backwards pass
                    self.debug_print(
                        f"RECEIVED BACKWARD: {round((data.__sizeof__() - 5) / 1e6, 1)} MB"
                    )

                    # Master-specific handling (ie for DistributedModel)
                    if self.master:
                        [n_iter, n_micro, module_id], tensor = pickle.loads(data[8:])
                        self.modules["Master"].backward_queues[n_micro].put(
                            ([n_iter, n_micro, module_id], tensor)
                        )

                    # Module-specific handling (ie for OffloadedModule / nn.Module)
                    elif self.modules:
                        (n_iter, n_micro, module_id), tensor = pickle.loads(data[8:])
                        self.modules[module_id].backward_queues.put(
                            ([n_iter, n_micro], tensor)
                        )

                # Handle requests for module parameters
                elif b"PARAMS-REQ" == data[:10]:
                    self.debug_print(f"RECEIVED PARAMS REQUEST")

                    # TODO Must ensure requesting node is indeed the master or an overseeing validator
                    module_id = data[10:]
                    self.send_parameters(
                        node, self.modules[module_id].parameters(), module_id
                    )

                    return True

                # Handle and store responses from a parameters request
                elif b"PARAMETERS" == data[:10]:
                    self.debug_print(f"RECEIVED PARAMS REQUEST")
                    module_id, parameters = pickle.loads(data[10:])
                    self.parameters[module_id] = parameters

                elif b"MODULE" == data[:6]:
                    self.debug_print(
                        f"RECEIVED: {round((data.__sizeof__() - 5) / 1e6, 1)} MB"
                    )

                    module = pickle.loads(data[6:])
                    module.forward_queues = queue.Queue()
                    module.backward_queues = queue.Queue()
                    module.intermediates = {}
                    module.host = node.node_id

                    self.modules[module.id] = module
                    # self.optimizers[module.id] = optim.Adam(module.parameters())

                    self.debug_print(f"Loaded distributed module!")
                    self.send_to_node(node, b"LOADED" + module.id)

                else:
                    # We do not log a ghost here since SmartNode is meant to be a super class and this should
                    # only be invoked by a super call
                    return False

            if ghost > 0:
                self.update_node_stats(node.node_id, "GHOST")
                # TODO: potentially some form of reporting mechanism via ip and port

            return True

        except Exception as e:
            self.debug_print(f"handle_data: Error handling data: {e}")

    def handle_requests(self, request=None):
        """Handles interactions between model and node processes"""
        if request is None:
            request = self.request_queue.get()

        req_type = request["type"]

        if req_type == "get_connection":
            node_id = request["args"]
            node = self.nodes[node_id]
            self.response_queue.put({"status": "SUCCESS", "return": node})

        elif req_type == "send_model":
            model, worker_id = request["args"]
            node = self.nodes[worker_id]
            self.send_module(model, node)

            while b"MODULE" in self.requests[node.node_id]:
                time.sleep(1)

            self.response_queue.put({"status": "SUCCESS", "return": None})

        elif req_type == "send_forward":
            worker_id, (args, kwargs, tag) = request["args"]
            node = self.nodes[worker_id]
            self.send_forward(node, (args, kwargs), tag)
            self.response_queue.put({"status": "SUCCESS", "return": None})

        elif req_type == "release_memory":
            key = request["args"]
            self.memory_manager[key].close()
            self.memory_manager[key].unlink()
            self.response_queue.put({"status": "SUCCESS", "return": None})

        elif req_type == "connect_node":
            node_id, host, port = request["args"]
            connected = self.connect_node(node_id, host, port)
            self.response_queue.put({"status": "SUCCESS", "return": connected})

        elif req_type == "info":
            self.response_queue.put({"status": "SUCCESS", "return": (self.rsa_key_hash, self.host, self.port)})

        elif req_type == "stop":
            self.response_queue.put({"status": "SUCCESS", "return": None})
            self.stop()

    def send_forward(self, node: Connection, args, context):
        """Send forward pass to node, must contain args (module args) and context (module + epoch id)"""
        pickled_data = b"FORWARD" + pickle.dumps((context, args))
        # self.store_request(node.node_id, )
        self.send_to_node(node, pickled_data)

    def store_tensor_in_shared_memory(self, key, tensor):
        id_hash = self.get_module_hash_from_id(key[2])
        tensor_out = handle_output(tensor)
        tensor_shape = tensor_out.shape
        tensor_dtype = tensor_out.dtype
        del tensor_out

        tensor = pickle.dumps(tensor)
        size = len(tensor)

        shm = shared_memory.SharedMemory(create=True, size=size)
        buffer = shm.buf[:size]
        buffer[:] = tensor

        self.modules[id_hash]["forward_queue"][key] = (tensor_shape, size, tensor_dtype, shm.name)
        self.memory_manager[key] = shm

    def send_backward(self, node: Connection, args, context):
        """Send backward pass to node, must contain args (module args) and context (module + epoch id)"""
        pickled_data = b"BACKWARD" + pickle.dumps((context, args))
        self.send_to_node(node, pickled_data)

    def send_parameters(self, node: Connection, parameters, module_id):
        """Send specific module parameters
        TODO should be accompanied by a requested proof (from smart contract) or the specific user
        """
        pickled_data = b"PARAMETERS" + pickle.dumps((module_id, list(parameters)))
        self.send_to_node(node, pickled_data)

    def send_parameters_req(self, node: Connection, module_id):
        """Request parameters from a specific worker"""
        self.send_to_node(node, b"PARAMS-REQ" + module_id)

    def send_module(self, module: nn.Module, node: Connection):
        module_bytes = pickle.dumps(module)
        self.debug_print(f"Sending module: {len(module_bytes)} to worker: {node.node_id}")
        self.store_request(node.node_id, b"MODULE")
        self.send_to_node(node, b"MODULE" + module_bytes)

    def listen_requests(self):
        while not self.terminate_flag.is_set():
            self.handle_requests()

    def get_module_hash_from_id(self, mod_id: bytes):
        for mod_hash in self.modules:
            if str(self.modules[mod_hash]["mod_id"]).encode() == mod_id:
                return mod_hash
        return None

    # def run(self):
    #     # Accept users and back-check history
    #     # Get proposees from SC and send our state to them
    #     # If we are the next proposee, accept info from validators and only add info to the final state if there are
    #     # 2 or more of the identical info
    #     listener = threading.Thread(target=self.listen, daemon=True)
    #     listener.start()
    #
    #     mp_comms = threading.Thread(target=self.listen_requests, daemon=True)
    #     mp_comms.start()
    #
    #     while not self.terminate_flag.is_set():
    #         # Handle job oversight, and inspect other jobs (includes job verification and reporting)
    #         pass
    #
    #     print("Node stopping...")
    #     for node in self.nodes.values():
    #         node.stop()
    #
    #     for node in self.nodes.values():
    #         node.join()
    #
    #     listener.join()
    #     mp_comms.join()
    #
    #     self.sock.settimeout(None)
    #     self.sock.close()
    #     print("Node stopped")
