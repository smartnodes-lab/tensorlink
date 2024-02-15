from src.p2p.smart_node import SmartNode
from src.p2p.connection import Connection
from src.ml.model_analyzer import estimate_memory, handle_output
from src.ml.distributed import DistributedModel

import torch.nn as nn
import threading
import pickle
import torch
import queue
import time
import os


class DistributedModule(nn.Module):
    def __init__(self, master_node, worker_node: Connection):
        super().__init__()
        self.master_node = master_node
        self.worker_node = worker_node
        # self.event = threading.Event()

    def forward(self, *args, **kwargs):
        self.master_node.send_forward(self.worker_node, (args, kwargs))

        # Must somehow wait for the response output from the worker
        time.sleep(3)

        if self.master_node.forward_relays.not_empty:
            return self.master_node.forward_relays.get()

    def backward(self, *args, **kwargs):

        # Must somehow get the response output from the worker
        self.master_node.send_tensor((args, kwargs), self.worker_node)


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
    else:
        # CPU should be able to handle 1 GB (temporary fix)
        memory += 0.4e9

    return memory


class Worker(SmartNode):
    """
    Todo:
        - confirm workers public key with smart contract
        - convert pickling to json for security (?)
        - process other jobs/batches while waiting for worker response (?)
        - link workers to database for complete offloading
        - different subclasses of Worker for memory requirements to designate memory-specific
            tasks, ie distributing a model too large to handle on a single computer / user
        - function that detaches the huggingface wrapped outputs tensor without modifying the rest
    """
    def __init__(self, host: str, port: int, wallet_address: str, debug: bool = False, max_connections: int = 0):
        super(Worker, self).__init__(
            host, port, wallet_address, debug=debug, max_connections=max_connections, callback=self.stream_data
        )

        # Model training parameters
        self.training = False
        self.master = False
        self.available_memory = get_gpu_memory()
        self.nodes = []
        self.peer_stats = []

        # For storing forward, backward, and intermediate tensors
        # Should be switched to some other data structure that relates to specific epochs
        self.forward_relays = queue.Queue()
        self.backward_relays = queue.Queue()
        self.intermediates = []
        self.model = None
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
            if b"DONE STREAM" == data[:11]:
                file_name = f"streamed_data_{node.host}_{node.port}"

                with open(file_name, "rb") as f:
                    streamed_bytes = f.read()

                self.stream_data(streamed_bytes, node)

                os.remove(file_name)

            elif b"FORWARD" == data[:7]:
                self.debug_print(f"RECEIVED FORWARD")
                if self.master or (self.training and self.model):
                    pickled = pickle.loads(data[7:])
                    self.forward_relays.put(pickled)

            elif b"BACKWARD" == data[:8]:
                self.debug_print(f"RECEIVED BACKWARD")
                if self.master or (self.training and self.model):
                    pickled = pickle.loads(data[8:])
                    self.backward_relays.put(pickled)

            elif b"PoL" == data[:3]:
                self.debug_print(f"RECEIVED PoL REQUEST")
                if self.training and self.model:
                    dummy_input = pickle.loads(data[3:])

                    proof_of_learning = self.proof_of_learning(dummy_input)

                    self.send_to_node(node, proof_of_learning)

            elif b"TENSOR" == data[:6]:
                if self.training:
                    tensor = pickle.loads(data[6:])

                    # Confirm identity/role of node
                    if node in self.inbound:
                        self.forward_relays.put(tensor)
                    elif node in self.outbound:
                        self.backward_relays.put(tensor)

            elif b"MODEL" == data[:5]:
                self.debug_print(f"RECEIVED: {round((data.__sizeof__() - 5) / 1e6, 1)} MB")
                if self.training and not self.model:
                    # Load in model
                    pickled = pickle.loads(data[5:])
                    self.model = pickled
                    self.training = True
                    self.debug_print(f"Loaded submodule!")

            elif b"DMODEL" == data[:6]:
                self.debug_print(f"RECEIVED: {round((data.__sizeof__() - 5) / 1e6, 1)} MB")
                if self.training and not self.model:
                    # Load in model
                    pickled = pickle.loads(data[6:])
                    # self.request_statistics()
                    time.sleep(0.5)
                    self.model = DistributedModel(self, pickled, self.peer_stats)
                    self.training = True
                    self.debug_print(f"Loaded distributed module!")

        except Exception as e:
            self.debug_print(f"worker:stream_data:{e}")
            raise e

    def run(self):
        # Thread for handling incoming connections
        listener = threading.Thread(target=self.listen, daemon=True)
        listener.start()

        # # Thread for handling incoming tensors from connected nodes
        # data_stream = threading.Thread(target=self.stream_data, daemon=True)
        # data_stream.start()

        # Main worker loop
        while not self.terminate_flag.is_set():
            if self.training and self.port != 5026:  # Port included for master testing without master class
                self.train_loop()

            # Include the following steps:
            # 1. Broadcast GPU memory statistics
            # self.broadcast_statistics()

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
        if self.backward_relays.empty() is False:
            next_node = self.outbound[0]  # Placeholder for the connecting node

            # Grab backwards pass from forward node and our associated input/output from forward pass
            loss_relay = self.backward_relays.get()
            assoc_input, assoc_output = self.intermediates.pop(-1)

            # Continue backwards pass on our section of model
            assoc_output.backward(loss_relay, retain_graph=True)  # Do we need retain graph?

            # self.optimizer.zero_grad()
            # self.optimizer.step()

            dvalues = assoc_input.grad

            # Pass along backwards pass to next node
            self.send_backward(next_node, dvalues)

        # Complete any forward pass
        if self.forward_relays.empty() is False:
            next_node = self.inbound[0]  # Placeholder for the appropriate node
            prev_forward = self.forward_relays.get()  # Grab queued forward pass unpack values (eg. mask, stride...)

            if isinstance(prev_forward, tuple):
                args, kwargs = prev_forward
            else:
                args = prev_forward
                kwargs = {}

            # Clear tensor of any previous info, set up for custom backward pass
            inp = handle_output(args).clone().detach().requires_grad_()  # This should be done on sending node, not receiving
            out = self.model(inp, **kwargs)

            # Store output and input tensor for backward pass
            self.intermediates.append([inp, handle_output(out)])

            # Relay forward pass to the next node
            self.send_forward(next_node, out)

    def proof_of_learning(self, dummy_input: torch.Tensor):
        proof = {
            "node_id": self.name,
            "memory": self.available_memory,
            "learning": self.training,
            "model": self.model,
        }

        if self.training:
            proof["output"] = handle_output(self.model(dummy_input)).sum()

    def send_tensor(self, tensor, node: Connection):
        # tensor_bytes = self.BoT + pickle.dumps(tensor) + self.EoT
        tensor_bytes = b"TENSOR" + pickle.dumps(tensor)
        self.debug_print(f"worker: sending {round(tensor_bytes.__sizeof__() / 1e9, 3)} GB")
        self.send_to_node(node, tensor_bytes)

    def send_forward(self, node: Connection, args):
        pickled_data = b"FORWARD" + pickle.dumps(args)
        self.send_to_node(node, pickled_data)

    def send_backward(self, node: Connection, args):
        pickled_data = b"BACKWARD" + pickle.dumps(args)
        self.send_to_node(node, pickled_data)

    def send_module(self, module: nn.Module, node: Connection, prefix: bytes = b""):
        module_bytes = pickle.dumps(module)
        module_bytes = prefix + b"MODEL" + module_bytes
        print("SENDING MODULE")
        self.send_to_node(node, module_bytes)
        time.sleep(1)

    # def host_job(self, model: nn.Module):
    #     """
    #     Todo:
    #         - connect to master node via SC and load in model
    #         - attempt to assign and relay model to other idle connected workers
    #         - determine relevant connections
    #     """
    #     self.optimizer = torch.optim.Adam
    #     self.training = True
    #     self.distribute_model(model)  # Add master vs worker functionality

    """Key Methods to Implement"""
    def get_jobs(self):
        pass
        # Confirm job details with smart contract, receive initial details from a node?

    def broadcast_statistics(self):
        memory = str(get_gpu_memory())

        if self.training:
            pass
            # Incorporate proofs of training, will be moved to proof_of_learning.py
            # proof1 = self.proof_of_model()
            # proof2 = self.proof_of_optimization()
            # proof3 = self.proof_of_output()

        self.send_to_nodes(memory.encode())

    def update_statistics(self):
        self.peer_stats = [{"id": i, "memory": 0.5e9, "connection": self.outbound[i],
                            "latency_matrix": self.outbound[i].latency} for i in range(len(self.outbound))]
