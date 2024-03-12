from src.p2p.connection import Connection
from src.p2p.smart_node import SmartNode
from src.ml.model_analyzer import get_gpu_memory

import torch.nn as nn
import threading
import hashlib
import pickle
import time


class TorchNode(SmartNode):
    def __init__(self, host: str, port: int, wallet_address: str, debug: bool = False, max_connections: int = 0,
                 callback=None):
        super(TorchNode, self).__init__(
            host, port, wallet_address, debug=debug, max_connections=max_connections, callback=callback
        )
        # State info
        self.available_memory = get_gpu_memory()
        self.state = 0

        # Stores connections and their context
        self.nodes = {}
        self.distributed_graph = {}
        self.updater_flag = threading.Event()

        self.key_hash = hashlib.sha256(self.rsa_pub_key).hexdigest()

    def send_tensor(self, tensor, node: Connection):
        # tensor_bytes = self.BoT + pickle.dumps(tensor) + self.EoT
        tensor_bytes = b"TENSOR" + pickle.dumps(tensor)
        self.debug_print(f"worker: sending {round(tensor_bytes.__sizeof__() / 1e9, 3)} GB")
        self.send_to_node(node, tensor_bytes)

    def send_forward(self, node: Connection, args, context: bytes = b""):
        pickled_data = b"FORWARD" + context + pickle.dumps(args)
        self.send_to_node(node, pickled_data)

    def send_backward(self, node: Connection, args, context: bytes = b""):
        pickled_data = b"BACKWARD" + context + pickle.dumps(args)
        self.send_to_node(node, pickled_data)

    def send_module(self, module: nn.Module, node: Connection, prefix: bytes = b""):
        module.parent = self.key_hash
        module_bytes = pickle.dumps(module)
        module_bytes = prefix + b"MODULE" + module_bytes

        print("SENDING MODULE")
        self.send_to_node(node, module_bytes)
        time.sleep(1)

    def send_statistics_request(self, worker_node):
        message = b"REQUEST"
        self.send_to_node(worker_node, message)

    def broadcast_statistics(self, callee, additional_context: dict = None):
        memory = self.available_memory

        stats = {"id": self.rsa_pub_key + self.port.to_bytes(4, "big"), "memory": memory, "state": self.state}

        if additional_context is not None:
            for k, v in additional_context.items():
                if k not in stats.keys():
                    stats[k] = v

        stats_bytes = pickle.dumps((self.key_hash, stats))
        stats_bytes = b"RESPONSE" + stats_bytes
        self.send_to_node(callee, stats_bytes)

    # Iterate connected nodes and request their current state
    def update_worker_stats(self):
        while not self.updater_flag.is_set():

            for node in self.connections:
                # Beforehand, check the last time the worker has updated (self.prune_workers?)
                self.send_statistics_request(node)
                time.sleep(1)

            if self.nodes:
                self.updater_flag.set()

            time.sleep(5)

    def select_candidate_worker(self):
        candidate_node = max(self.nodes.values(), key=lambda x: x["memory"])
        return candidate_node

    # Potential separate streaming for universal data transmission
    # def torch_stream(self):
