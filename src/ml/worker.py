from src.p2p.linked_node import LinkedNode
from src.p2p.connection import Connection

from transformers import BertModel
import torch.nn as nn
import threading
import random
import socket
import torch
import queue
import time
import io
import os


def get_gpu_memory():
    # Check how much available memory we can allocate to the node
    memory = 0

    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
        memory += torch.cuda.memory

        for device in devices:
            torch.cuda.set_device(device)
            memory_stats = torch.cuda.memory_stats(device)
            device_memory = memory_stats["allocated_bytes.all.peak"] / 1024 / 1024
            memory += device_memory

    return memory


def get_first_layer(model: nn.Module):
    if len(list(model.children())) > 0:
        _, submodule = next(model.named_children())
        return get_first_layer(submodule)
    else:
        return model


def load_module(module_bytes: bytes):
    tensor = torch.load(io.BytesIO(module_bytes))
    return tensor


class Worker(LinkedNode):
    """
    model:
    - neighbouring and master node connections (including their relevant parameters + reputation?)
    - relevant data pertaining to worker's proof of work

    TODO:
    - confirm workers public key with smart contract
    """
    def __init__(self, host: str, port: int, debug: bool = False, max_connections: int = 0,
                 url: str = "wss://ws.test.azero.dev"):
        super(Worker, self).__init__(host, port, debug, max_connections, url)

        # Model training parameters
        self.training = False

        self.forward_batches = queue.Queue()
        self.backward_batches = queue.Queue()

        self.previous_node = None
        self.following_node = None

        self.model = None
        self.optimizer = None
        self.loss = nn.MSELoss()

    def stream_data(self):
        pass

    def run(self):
        # Thread for handling incoming connections
        listener = threading.Thread(target=self.listen, daemon=True)
        listener.start()

        # # Thread for handling incoming data
        # data_stream = threading.Thread(target=self.stream_data, daemon=True)
        # data_stream.start()

        # Main worker loop
        while not self.terminate_flag.is_set():
            if self.training:
                if self.backward_batches.empty() is False:
                    # Grab backwards pass from previous node
                    assoc_forward, backward_batch = self.backward_batches.get()
                    loss = self.loss(assoc_forward, backward_batch)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

                    dvalues = get_first_layer(self.model).weight.grad.detach()
                    self.send_tensor(dvalues, self.previous_node)

                elif self.forward_batches.empty() is False:
                    forward_batch = self.forward_batches.get()
                    out = self.model(forward_batch)
                    self.send_tensor(out, self.following_node)

            # Include the following steps:
            # 1. Broadcast GPU memory statistics
            # self.broadcast_statistics()

            # 3. Load the required model (if applicable)
            # For example, you can call self.load_model(model) with the model received from another node.

            # 4. Send output to the following node
            # For example, you can call self.send_to_nodes(output_data.encode())

            # 5. Handle requests for proof of training
            # For example, you can call self.proof_of_optimization(), self.proof_of_output(), etc.

            self.reconnect_nodes()
            time.sleep(0.01)

        print("Node stopping...")
        for node in self.all_nodes:
            node.stop()

        time.sleep(1)

        for node in self.all_nodes:
            node.join()

        self.sock.settimeout(None)
        self.sock.close()
        print("Node stopped")

    def send_tensor(self, tensor: torch.Tensor, node: Connection):
        torch.save(tensor, "tensor.pt")

        with open("tensor.pt", "rb") as f:
            tensor_bytes = f.read()

        self.send_to_node(node, tensor_bytes)
        # Handle tensor files post-send

    def initialize_job(self):
        """
        Todo:
            - connect to master node via SC and load in model
            - attempt to assign and relay model to other idle connected workers
            - determine relevant connections
        """
        self.model = nn.Sequential(
            nn.Linear(10, 1024),
            nn.Linear(1024, 1024)
        ) if self.port == 5026 else nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(10, 1024)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.previous_node = 0
        self.following_node = 1
        self.training = True

        # Dummy model split in 2, preset optimizer, static node determination for testing
        submodule = nn.Sequential(
            nn.Linear(10, 1024),
            nn.Linear(1024, 1024))

        optimizer = torch.optim.Adam(submodule.parameters())

        self.load_job(submodule, optimizer)

    def proof_of_optimization(self):
        pass

    def proof_of_output(self):
        pass

    def proof_of_model(self):
        pass

    def get_jobs(self):
        pass
        # Confirm job details with smart contract, receive initial details from a node?

    def broadcast_statistics(self):
        memory = str(get_gpu_memory())

        if self.training:
            # Incorporate proofs of training
            proof1 = self.proof_of_model()
            proof2 = self.proof_of_optimization()
            proof3 = self.proof_of_output()

        self.send_to_nodes(memory.encode())
