from src.p2p.smart_node import SmartNode
from src.p2p.connection import Connection
from src.ml.model_analyzer import estimate_memory, get_first_layer

import torch.nn as nn
import threading
import random
import socket
import pickle
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


class Worker(SmartNode):
    """
    model:
    - neighbouring and master node connections (including their relevant parameters + reputation?)
    - relevant data pertaining to worker's proof of work

    Todo:
        - confirm workers public key with smart contract
    """
    def __init__(self, host: str, port: int, debug: bool = False, max_connections: int = 0,
                 url: str = "wss://ws.test.azero.dev"):
        super(Worker, self).__init__(host, port, debug, max_connections, url, self.stream_data)

        # Model training parameters
        self.training = False
        self.master = False

        self.forward_batches = queue.Queue()
        self.backward_batches = queue.Queue()

        self.modules = []
        self.optimizer = None
        self.loss = None

        # Data transmission variables
        self.BoT = 0x00.to_bytes(1, 'big')
        self.EoT = 0x01.to_bytes(1, 'big')

    def stream_data(self, data: bytes, node: Connection):
        """
        Handle incoming tensors from connected nodes and new job requests

        Todo:
            - ensure correct nodes sending data
            - track position of tensor in the training session
            - potentially move/forward method directly to Connection to save data via identifying data
                type and relevancy as its streamed in, saves bandwidth if we do not need the data / spam
        """
        try:
            if b"TENSOR" == data[:6]:
                if self.training:
                    tensor = pickle.loads(data[6:])

                    # Confirm identity/role of node
                    if node in self.inbound:
                        self.forward_batches.put(tensor)
                    elif node in self.outbound:
                        self.backward_batches.put(tensor)

            elif b"MODEL_REQUEST" == data[:13]:
                if self.training is False:
                    # Load in model
                    pickled = pickle.loads(data[13:])
                    self.modules.append(pickled)
                    self.training = True

        except Exception as e:
            self.debug_print(f"worker:stream_data:{e}")

    def run(self):
        # Thread for handling incoming connections
        listener = threading.Thread(target=self.listen, daemon=True)
        listener.start()

        # # Thread for handling incoming tensors from connected nodes
        # data_stream = threading.Thread(target=self.stream_data, daemon=True)
        # data_stream.start()

        # Main worker loop
        while not self.terminate_flag.is_set():
            if self.training:
                self.train_loop()

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

    def train_loop(self):
        # Complete any outstanding back propagations
        if self.backward_batches.empty() is False:
            # Grab backwards pass from forward node and our associated forward pass output
            assoc_forward, backward_batch = self.backward_batches.get()

            # Continue backwards pass on our section of model
            loss = assoc_forward.backward(backward_batch, retain_graph=True)  # Do we need retain graph?
            self.optimizer.zero_grad()
            self.optimizer.step()

            # Pass along backwards pass to next node
            dvalues = get_first_layer(self.model).weight.grad.detach()
            self.send_tensor(dvalues, self.previous_node)
        # Complete any forward pass
        elif self.forward_batches.empty() is False:
            forward_batch = self.forward_batches.get()
            out = self.model(forward_batch)
            self.send_tensor(out, self.following_node)

    def send_tensor(self, tensor: torch.Tensor, node: Connection):
        # tensor_bytes = self.BoT + pickle.dumps(tensor) + self.EoT
        tensor_bytes = b"TENSOR" + pickle.dumps(tensor)
        self.debug_print(f"worker: sending {round(tensor_bytes.__sizeof__() / 1e9, 3)} GB")
        self.send_to_node(node, tensor_bytes)

    def send_module(self, module: nn.Module):
        pass

    def initialize_job(self, model: nn.Module):
        """
        Todo:
            - connect to master node via SC and load in model
            - attempt to assign and relay model to other idle connected workers
            - determine relevant connections
        """
        self.optimizer = torch.optim.Adam
        self.training = True
        self.distribute_model(model)  # Add master vs worker functionality

    def distribute_model(self, model: nn.Module):
        """
        Distribute model to connected nodes, assign modules based on memory requirements & latency
        """
        offloaded_memory = 0
        candidate_node = max(enumerate(node.memory for node in self.all_nodes), key=lambda x: x[1])[0]

        if len(list(model.children())) > 0:
            for name, submodule in model.named_children():
                submodule_memory_estimate = estimate_memory(submodule)
                offloaded_memory += submodule_memory_estimate

                self.send_module(submodule)

    """Key Methods to Implement"""
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
