from tensorlink.ml.utils import get_gpu_memory
from tensorlink.mpc.shared_memory import get_from_shared_memory
from tensorlink.p2p.connection import Connection
from tensorlink.p2p.smart_node import SmartNode

from multiprocessing import shared_memory
import logging
import pickle
import queue
import threading
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
        role,
        max_connections: int = 0,
        upnp=True,
        off_chain_test=False,
        local_test=False,
    ):
        super(TorchNode, self).__init__(
            role=role,
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
            local_test=local_test,
        )

        # Available GPU mpc estimation
        self.available_memory = get_gpu_memory()

        self._mpc_comms = None
        self.memory_manager = {}
        self.request_queue = request_queue
        self.response_queue = response_queue

        # Pointers to model parameters in DistributedModels
        self.modules = {}
        self.state_updates = {}

        # Master flag for handling different types of storage as master
        self.master = False
        self.mpc_terminate_flag = threading.Event()

    def handle_data(self, data: bytes, node: Connection):
        try:
            # Call parent class's handle_data, if applicable
            handled = super().handle_data(data, node)

            if not handled:
                # Define a dictionary mapping prefixes to handler methods
                handlers = {
                    b"LOADED": self._handle_module_loaded,
                    b"FORWARD": self._handle_forward,
                    b"BACKWARD": self._handle_backward,
                    b"OPTIMIZER-RESPONSE": self._handle_optimizer_response,
                    b"OPTIMIZER": self._handle_optimizer_request,
                    b"PARAMS-REQ": self._handle_parameters_request,
                    b"PARAMETERS": self._handle_parameters,
                    b"MODULE": self._handle_module,
                    b"UPDATE-TRAIN": self._update_train,
                    b"TRAIN-UPDATED": self._train_updated,
                }

                # Iterate through handlers to find the matching prefix
                for prefix, handler in handlers.items():
                    if data.startswith(prefix):
                        return (
                            handler(data, node)
                            if prefix
                            in (
                                b"LOADED",
                                b"FORWARD",
                                b"BACKWARD",
                                b"OPTIMIZER-RESPONSE",
                                b"OPTIMIZER",
                                b"MODULE",
                                b"UPDATE-TRAIN",
                            )
                            else handler(data)
                        )

                return False

        except Exception as e:
            self.debug_print(
                f"TorchNode -> Error handling data: {e}",
                colour="bright_red",
                level=logging.ERROR,
            )

    def _train_updated(self, data: bytes):
        mode = False if data[13:14] == b"0" else True
        module_id = data[14:78].decode()
        if module_id in self.modules:
            self.modules[module_id]["training"] = mode

    def _update_train(self, data: bytes, node: Connection):
        mode = False if data[12:13] == b"0" else True
        module_id = data[13:77].decode()
        self.modules[module_id]["training"] = mode
        self.send_train_updated(node, mode, module_id)

    def _handle_parameters(self, data: bytes):
        module_id = data[10:74].decode()
        self.debug_print(
            f"TorchNode -> Received Parameters for: {module_id}",
            colour="blue",
        )
        file_name = f"tmp/{module_id}_parameters"
        key = "P" + module_id
        self.memory_manager[key] = file_name
        return True

    def _handle_parameters_request(self, data: bytes):
        self.debug_print("TorchNode -> RECEIVED PARAMS REQUEST")

        # TODO Must ensure requesting roles is indeed the master or an overseeing validator
        module_id = data[10:74].decode()
        self.memory_manager["P" + module_id] = True
        return True

    def _handle_optimizer_request(self, data: bytes, node: Connection):
        if self.role == "V" or node.node_id not in self.nodes:
            node.ghosts += 1
            return False
        else:
            module_id, optimizer_fn, optimizer_kwargs = pickle.loads(data[9:])
            self.state_updates[module_id].append((optimizer_fn, optimizer_kwargs))
            return True

    def _handle_optimizer_response(self, data: bytes, node: Connection):
        if self.role == "V" or node.node_id not in self.nodes:
            node.ghosts += 1
            return False
        else:
            module_id, response_type = pickle.loads(data[18:])

            if response_type == "loaded":
                self.debug_print(
                    f"TorchNode -> Optimizer for module: {module_id} loaded on worker {node.node_id}",
                    level=logging.INFO,
                    colour="bright_cyan",
                )
            elif response_type == "stepped":
                self.debug_print(
                    f"TorchNode -> Optimizer for module: {module_id} stepped on worker {node.node_id}",
                    colour="bright_cyan",
                )
            elif response_type == "zeroed":
                self.debug_print(
                    f"TorchNode -> Optimizer for module: {module_id} zeroed on worker {node.node_id}",
                    colour="bright_cyan",
                )

            self.state_updates[module_id].append(response_type + node.node_id)
            return True

    def _handle_backward(self, data: bytes, node: Connection):
        # Basic check, must be upgraded to check if we are expecting the request
        if self.role == "V" or node.node_id not in self.nodes:
            node.ghosts += 1
            return False
        else:
            # Find size parameter within bytes
            eos = data.find(b"::")
            size = int(data[8:eos])

            formatted_size = format_size(size)
            self.debug_print(f"TorchNode -> RECEIVED BACKWARD: {formatted_size}")

            # TODO we must check that the forward received corresponds to a sent pass/specific module
            # must also do with backwards
            tensor = data[eos + 2 : eos + 2 + size]
            key = tuple(pickle.loads(data[eos + 2 + size :]))

            # Create shared mpc block and store tensor
            self._store_tensor_in_shared_memory(key, tensor, backward=True)
            return True

    def _handle_forward(self, data: bytes, node: Connection):
        # Basic check, must be upgraded to check if we are expecting the request
        if self.role == "V" or node.node_id not in self.nodes:
            node.ghosts += 1
            return False
        else:
            # Received a forward pass
            eos = data.find(b"::")
            size = int(data[7:eos])
            formatted_size = format_size(size)
            self.debug_print(f"TorchNode -> RECEIVED FORWARD: {formatted_size}")

            # TODO we must check that the forward received corresponds to a sent pass/specific module
            # must also do with backwards
            tensor = data[eos + 2 : eos + 2 + size]
            key = tuple(pickle.loads(data[eos + 2 + size :]))

            # Create shared mpc block and store tensor
            self._store_tensor_in_shared_memory(key, tensor)
            return True

    def _handle_module(self, data: bytes, node: Connection):
        module_id = data[6:70].decode()
        module_name = None
        optimizer_name = None
        request_to_remove = []

        if node.node_id in self.requests:
            for req in self.requests[node.node_id]:
                if module_id in req:
                    module_name = req[len(module_id) :]
                    request_to_remove.append(req)

                if "OPTIMIZER" in req:
                    optimizer_name = req[9:]
                    request_to_remove.append(req)

            for req in request_to_remove:
                self._remove_request(node.node_id, req)

            if module_name is not None:
                self.debug_print(f"TorchNode -> Received Module: {module_id}")

                self.modules[module_id] = {
                    "mem_info": module_id,
                    "host": node.node_id,
                    "forward_queue": {},
                    "backward_queue": {},
                    "name": module_name,
                    "optimizer": optimizer_name,
                }
                self.state_updates[module_id] = []

                self.debug_print(
                    "TorchNode -> Loaded distributed module!",
                    colour="bright_cyan",
                    level=logging.INFO,
                )
                return True

            else:
                node.ghosts += 1
        else:
            node.ghosts += 1

        return False

    def _handle_module_loaded(self, data: bytes, node: Connection):
        """Remove load module request to signal to distributed process"""
        self.debug_print(
            f"TorchNode -> Successfully offloaded submodule to: {node.node_id}",
            level=logging.INFO,
            colour="bright_cyan",
        )
        module_id = data[6:70].decode()
        self._remove_request(node.node_id, "MODULE" + module_id)
        return True

    def handle_requests(self, request=None):
        """Handles interactions between model and roles processes."""
        try:
            if request is None:
                try:
                    request = self.request_queue.get(timeout=3)
                except queue.Empty:
                    return

            req_type = request.get("type")
            if not req_type:
                self.response_queue.put(
                    {"status": "FAILURE", "error": "Invalid request type"}
                )
                return

            handlers = {
                "get_connection": self._handle_get_connection,
                "send_model": self._handle_send_model,
                "check_loaded": self._handle_check_module_loaded,
                "module_loaded": self._handle_module_loaded_request,
                "optimizer_response": self._handle_optimizer_response_request,
                "send_forward": self._handle_send_forward,
                "send_backward": self._handle_send_backward,
                "send_parameters": self._handle_send_parameters,
                "is_loaded": self._handle_is_loaded,
                "check_module": self._handle_check_module,
                "check_module_request": self._handle_check_module_request,
                "check_forward": self._handle_check_forward,
                "check_backward": self._handle_check_backward,
                "send_optimizer_request": self._handle_send_optimizer_request,
                "check_state_update": self._handle_check_state_update,
                "check_validators": self._handle_check_validators,
                "check_parameters_request": self._handle_check_parameters_request,
                "check_parameters": self._handle_check_parameters,
                "request_parameters": self._handle_request_parameters,
                "update_train": self._handle_update_train,
                "check_train": self._handle_check_train,
                "release_memory": self._handle_release_memory,
                "check_shutdown": self._handle_check_shutdown,
                "stop": self._handle_stop,
                "connect_node": self._handle_connect_node,
                "info": self._handle_get_info,
                "debug_print": self._handle_debug_print,
            }

            handler = handlers.get(req_type)

            if handler:
                handler(request)
            else:
                self.response_queue.put(
                    {"status": "FAILURE", "error": f"Unknown request type: {req_type}"}
                )
        except Exception as e:
            self.response_queue.put({"status": "FAILURE", "error": str(e)})

    def _handle_get_connection(self, request):
        # Get connection info from a roles id
        node_id = request["args"]
        node = self.nodes[node_id]
        self.response_queue.put({"status": "SUCCESS", "return": node})

    def _handle_send_model(self, request):
        # Send module that is stored in shared mpc to another roles
        name, worker_id, module_id = request["args"]
        node = self.nodes[worker_id]
        node.adjust_chunk_size("large")
        self.send_module(name, module_id, node)
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_check_module_loaded(self, request):
        # Check if sent module has been received and loaded on the other nodes
        worker_id, module_id = request["args"]
        return_val = False

        if "MODULE" + module_id not in self.requests[worker_id]:
            return_val = True

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_module_loaded_request(self, request):
        # Send module loaded message to roles
        module_id = request["args"]
        node_id = self.modules[module_id]["host"]
        node = self.nodes[node_id]
        self.send_to_node(node, b"LOADED" + module_id.encode())
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_optimizer_response_request(self, request):
        module_id, response_type = request["args"]
        node_id = self.modules[module_id]["host"]
        node = self.nodes[node_id]

        self.send_to_node(
            node,
            b"OPTIMIZER-RESPONSE" + pickle.dumps((module_id, response_type)),
        )

        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_send_forward(self, request):
        # Send forward pass tensor from shared mpc to a roles
        worker_id, size, shm_name, tag = request["args"]
        node = self.nodes[worker_id]
        forward_bytes = get_from_shared_memory(size, shm_name, encoded=True)
        self.send_forward(node, forward_bytes, tag)
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_send_backward(self, request):
        # Send backwards pass from shared mpc to a roles
        worker_id, size, shm_name, tag = request["args"]
        node = self.nodes[worker_id]
        backward_bytes = get_from_shared_memory(size, shm_name, encoded=True)
        self.send_backward(node, backward_bytes, tag)
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_send_parameters(self, request):
        node_id, module_id = request["args"]
        node = self.nodes[node_id]
        self.send_to_node_from_file(
            node, f"parameters_{module_id}", b"PARAMETERS" + module_id.encode()
        )
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_is_loaded(self, request):
        return_val = False
        for module_id, module in self.modules.items():
            if module.get("terminated"):
                pass
            else:
                return_val = True

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_module(self, request):
        # Check if module has been received and is loaded in shared mpc
        return_val = False
        for module_id, module in self.modules.items():
            if "mem_info" in module:
                name = module["mem_info"]
                return_val = (
                    name,
                    module_id,
                    module["host"],
                    module["name"],
                    module["optimizer"],
                )
                del module["mem_info"]
            elif "termination" in module:
                return_val = module_id
                del module["termination"]
                module["terminated"] = True

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_module_request(self, request):
        request_type, worker_id, module_id = request["args"]
        return_val = False
        key = None

        if module_id in self.state_updates.keys():
            key = request_type + worker_id

        if key in self.state_updates[module_id]:
            self.state_updates[module_id].remove(key)
            return_val = True

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_forward(self, request):
        # Check if forward pass has been received and is loaded in shared mpc
        return_val = None

        if self.role == "W":
            module_id = request["args"]

            if module_id in self.modules:
                module = self.modules[module_id]
                min_iter, min_micro = -1, -1
                for n_iter, n_micro, module_id in module["forward_queue"].keys():
                    if n_iter <= min_iter or min_iter == -1:
                        min_iter = n_iter
                    if n_micro <= min_micro or min_micro == -1:
                        min_micro = n_micro

                key = (min_iter, min_micro, module_id)

                if key in module["forward_queue"]:
                    return_val = (key, module["forward_queue"][key])
                    del module["forward_queue"][key]

        else:
            n_iter, n_micro, module_id = request["args"]

            if module_id in self.modules:
                if request["args"] in self.modules[module_id]["forward_queue"]:
                    return_val = self.modules[module_id]["forward_queue"][
                        request["args"]
                    ]
                    del self.modules[module_id]["forward_queue"][request["args"]]

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_backward(self, request):
        # Check if backward pass has been received and is loaded in shared mpc
        args = request["args"]
        return_val = None

        if self.role == "W":
            module_hash = args
            module = self.modules[module_hash]
            min_iter, min_micro = -1, -1
            for n_iter, n_micro, module_id in module["backward_queue"].keys():
                if n_iter <= min_iter or min_iter == -1:
                    min_iter = n_iter
                if n_micro <= min_micro or min_micro == -1:
                    min_micro = n_micro

            key = (min_iter, min_micro, module_hash)

            if key in module["backward_queue"]:
                return_val = (key, module["backward_queue"][key])
                del module["backward_queue"][key]

        else:
            n_iter, n_micro, module_hash, module_id = args
            key = (n_iter, n_micro, module_id)
            if module_hash in self.modules:
                if key in self.modules[module_hash]["backward_queue"]:
                    return_val = self.modules[module_hash]["backward_queue"][key]
                    del self.modules[module_id]["backward_queue"][key]

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_send_optimizer_request(self, request):
        worker_id, module_id, optimizer_fn, optimizer_kwargs = request["args"]
        node = self.nodes[worker_id]
        data = pickle.dumps((module_id, optimizer_fn, optimizer_kwargs))
        self.send_to_node(node, b"OPTIMIZER" + data)
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_check_state_update(self, request):
        module_id = request["args"]
        return_val = None
        if self.state_updates[module_id]:
            return_val = self.state_updates[module_id].pop()
        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_validators(self, request):
        return_val = len(self.validators)
        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_parameters_request(self, request):
        key = "P" + request["args"]
        return_val = None

        if key in self.memory_manager:
            del self.memory_manager[key]
            return_val = True
        else:
            return_val = False

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_check_parameters(self, request):
        module_id = request["args"]
        key = "P" + module_id
        if key in self.memory_manager:
            file_name = self.memory_manager[key]
            return_val = file_name
        else:
            return_val = None

        self.response_queue.put({"stats": "SUCCESS", "return": return_val})

    def _handle_request_parameters(self, request):
        worker_id, module_id = request["args"]
        node = self.nodes[worker_id]
        self.send_parameters_req(node, module_id)
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_update_train(self, request):
        worker_id, mode, module_id = request["args"]
        mode = b"0" if mode is False else b"1"
        node = self.nodes[worker_id]
        self.send_to_node(node, b"UPDATE-TRAIN" + mode + module_id.encode())
        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_check_train(self, request):
        module_id = request["args"]
        return_val = None

        if module_id in self.modules:
            if "training" in self.modules[module_id].keys():
                return_val = self.modules[module_id]["training"]

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_release_memory(self, request):
        data_type, module_id, key = tuple(request["args"])
        del self.memory_manager[key]
        if key in self.modules[module_id][data_type]:
            del self.modules[module_id][data_type][key]

        self.response_queue.put({"status": "SUCCESS", "return": None})

    def _handle_connect_node(self, request):
        node_id, host, port = request["args"]
        connected = self.connect_node(node_id, host, port)
        self.response_queue.put({"status": "SUCCESS", "return": connected})

    def _handle_get_info(self, request):
        self.response_queue.put(
            {
                "status": "SUCCESS",
                "return": (self.rsa_key_hash, self.host, self.port),
            }
        )

    def _handle_stop(self, request):
        self.response_queue.put({"status": "SUCCESS", "return": True})
        self.terminate_flag.set()

    def _handle_check_shutdown(self, request):
        if self.terminate_flag.is_set():
            self.response_queue.put({"status": "SUCCESS", "return": True})
            t = threading.Thread(target=self._stop_mpc_comms)
            t.start()
        else:
            self.response_queue.put({"status": "SUCCESS", "return": False})

    def _handle_debug_print(self, request):
        if len(request["args"]) == 1:
            message = request["args"][0]
            colour = None
            level = logging.DEBUG
        else:
            message, colour, level = request["args"]
        self.debug_print(message, colour=colour, level=level)
        self.response_queue.put({"status": "SUCCESS", "return": False})

    def send_forward(self, node: Connection, forward_bytes, context):
        """Send forward pass to roles, must contain args (module args) and context (module + epoch id)"""
        size = str(len(forward_bytes)).encode() + b"::"
        pickled_data = b"FORWARD" + size + forward_bytes + pickle.dumps(context)
        self.send_to_node(node, pickled_data)

    def _store_tensor_in_shared_memory(self, key, tensor: bytes, backward=False):
        id_hash = key[2]
        size = len(tensor)

        shm = shared_memory.SharedMemory(create=True, size=size)
        buffer = shm.buf[:size]
        buffer[:] = tensor

        queue = "forward_queue" if not backward else "backward_queue"

        self.modules[id_hash][queue][key] = (size, shm.name)
        self.memory_manager[key] = shm.name
        del buffer
        shm.close()

    def store_parameters_in_shared_memory(self, key, parameters):
        module_id = key[1:]
        parameters = pickle.dumps(parameters)
        size = len(parameters)

        shm = shared_memory.SharedMemory(create=True, size=size)
        buffer = shm.buf[:size]
        buffer[:] = parameters

        self.modules[module_id]["parameters"][key] = (size, shm.name)
        self.memory_manager[key] = shm.name

    def send_backward(self, node: Connection, backward_bytes, context):
        """Send backward pass to roles, must contain args (module args) and context (module + epoch id)"""
        size = str(len(backward_bytes)).encode() + b"::"
        pickled_data = b"BACKWARD" + size + backward_bytes + pickle.dumps(context)
        self.send_to_node(node, pickled_data)

    def send_parameters_req(self, node: Connection, module_id: str):
        """Request parameters from a specific worker"""
        self.send_to_node(node, b"PARAMS-REQ" + module_id.encode())

    def send_train_updated(self, node: Connection, mode: bool, module_id: str):
        mode = b"0" if mode is False else b"1"
        self.send_to_node(node, b"TRAIN-UPDATED" + mode + module_id.encode())

    def send_module(self, file_name: bytes, module_id: str, node: Connection):
        self.debug_print(
            f"TorchNode -> Sending module: {module_id} to worker: {node.node_id}",
            level=logging.INFO,
            colour="bright_blue",
        )
        self._store_request(node.node_id, "MODULE" + module_id)
        self.state_updates[module_id] = []
        self.send_to_node_from_file(node, file_name, b"MODULE" + module_id.encode())

    def _listen_requests(self):
        while not self.mpc_terminate_flag.is_set():
            self.handle_requests()
            time.sleep(0.01)

    def get_module_hash_from_id(self, mod_id: bytes):
        for mod_hash in self.modules:
            if str(self.modules[mod_hash]["mod_id"]).encode() == mod_id:
                return mod_hash
        return None

    def run(self):
        super().run()
        self._mpc_comms = threading.Thread(target=self._listen_requests, daemon=True)
        self._mpc_comms.start()

    def stop(self):
        super().stop()

    def _stop_mpc_comms(self):
        self.mpc_terminate_flag.set()
        self.debug_print(
            "Shutting down distributed ML processes...", level=logging.DEBUG
        )
        self._mpc_comms.join()
