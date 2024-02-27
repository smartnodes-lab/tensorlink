from src.p2p.connection import Connection
from src.p2p.smart_node import SmartNode
from src.ml.distributed import get_gpu_memory

import torch.nn as nn
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
        self.nodes = {}

    def send_tensor(self, tensor, node: Connection):
        # tensor_bytes = self.BoT + pickle.dumps(tensor) + self.EoT
        tensor_bytes = b"TENSOR" + pickle.dumps(tensor)
        self.debug_print(f"worker: sending {round(tensor_bytes.__sizeof__() / 1e9, 3)} GB")
        self.send_to_node(node, tensor_bytes)

    def send_forward(self, node: Connection, context, args):
        pickled_data = b"FORWARD" + context + pickle.dumps(args)
        self.send_to_node(node, pickled_data)

    def send_backward(self, node: Connection, context, args):
        pickled_data = b"BACKWARD" + pickle.dumps(args)
        self.send_to_node(node, pickled_data)

    def send_module(self, module: nn.Module, node: Connection, prefix: bytes = b""):
        module_bytes = pickle.dumps(module)
        module_bytes = prefix + b"MODEL" + module_bytes
        print("SENDING MODULE")
        self.send_to_node(node, module_bytes)
        time.sleep(1)

    def send_statistics_request(self, worker_node):
        message = b"REQUEST"
        print("request sent")
        self.send_to_node(worker_node, message)

    def broadcast_statistics(self, callee, additional_context: dict = None):
        memory = self.available_memory

        stats = {"id": self.rsa_pub_key, "memory": memory, "state": self.state}

        if additional_context is not None:
            for k, v in additional_context.items():
                if k not in stats.keys():
                    stats[k] = v

        key_hash = hashlib.sha256(self.rsa_pub_key).hexdigest()

        stats_bytes = pickle.dumps((key_hash, stats))
        stats_bytes = b"RESPONSE" + stats_bytes
        self.send_to_node(callee, stats_bytes)

    # Iterate connected nodes and request their current state
    def update_worker_stats(self):
        while not self.terminate_flag.is_set():
            for node in self.connections:
                # Beforehand, check the last time the worker has updated (self.prune_workers?)
                self.send_statistics_request(node)
                time.sleep(1)

            time.sleep(5)

    # Potential separate streaming for universal data transmission
    # def torch_stream(self):
