from src.p2p.smart_node import SmartNode
from src.p2p.connection import Connection
from src.ml.model_analyzer import estimate_memory, handle_output
from src.ml.distributed import DistributedModel, get_gpu_memory

import torch.nn as nn
import threading
import random
import pickle
import torch
import queue
import time
import uuid
import os


COLOURS = ['\033[91m', '\033[92m', '\033[93m', '\033[95m', '\033[94m']
RESET = '\033[0m'


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
        self.validator_nodes = []
        self.worker_nodes = []

        # For storing forward, backward, and intermediate tensors
        # Should be switched to some other data structure that relates to specific epochs
        self.forward_relays = queue.Queue()
        self.backward_relays = queue.Queue()
        self.intermediates = []

        self.model = None
        self.optimizer = None
        self.loss = None

    def stream_data(self, data: bytes, node: Connection):
        """
        Handle incoming tensors from connected nodes and new job requests

        Todo:
            - ensure correct nodes sending data
            - potentially move/forward method directly to Connection to save data via identifying data
                type and relevancy as its streamed in, saves bandwidth if we do not need the data / spam
        """

        try:
            # The case where we load via downloaded pickle file (potential security threat)
            if b"DONE STREAM" == data[:11]:
                file_name = f"streamed_data_{node.host}_{node.port}"

                with open(file_name, "rb") as f:
                    streamed_bytes = f.read()

                self.stream_data(streamed_bytes, node)

                os.remove(file_name)

            elif b"FORWARD" == data[:7]:
                self.debug_print(f"RECEIVED FORWARD: {round((data.__sizeof__() - 5) / 1e6, 1)} MB")
                if self.master or (self.training and self.model):
                    pickled = pickle.loads(data[7:])
                    self.forward_relays.put(pickled)

            elif b"BACKWARD" == data[:8]:
                self.debug_print(f"RECEIVED BACKWARD: {round((data.__sizeof__() - 5) / 1e6, 1)} MB")
                if self.master or (self.training and self.model):
                    pickled = pickle.loads(data[8:])
                    self.backward_relays.put(pickled)

            # elif b"PoL" == data[:3]:
            #     self.debug_print(f"RECEIVED PoL REQUEST")
            #     if self.training and self.model:
            #         dummy_input = pickle.loads(data[3:])
            #
            #         proof_of_learning = self.proof_of_learning(dummy_input)
            #
            #         self.send_to_node(node, proof_of_learning)

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
                #
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
        # data_stream = threading.Thread(target=self.stream_data, daemo=True)
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

    """Key Methods to Implement"""
    def distribute_model(self, model: nn.Module):
        # Retrieve model names and associated offloaded workers. Contact candidate workers
        # and ensure they are ready to receive the model / train
        def recurse_model(module, nodes, candidate_node=None):
            # Get module information
            module_memory = estimate_memory(module)
            module_children = list(module.named_children())
            module_name = f"{type(module)}".split(".")[-1].split(">")[0][:-1]

            # See if we can handle the input first
            if module_memory < self.available_memory:
                self.available_memory -= module_memory
                return nodes, f"MASTER"

            elif module_memory < max(nodes, key=lambda x: x["memory"])["memory"]:
                candidate_node = max(enumerate(nodes), key=lambda x: x[1]["memory"])[0]
                nodes[candidate_node]["memory"] -= module_memory
                return nodes, f"{nodes[candidate_node]['id']}"

            # We cannot handle the module on the master node, there are now three options:
            # assume intermediate DistributedModel on master if graph is empty (or user wants to maximize his compute
            # contribution before offloading?)
            # attempt to offload module on best candidate
            # attempt intermediate DistributedModel on best candidate
            # elif not distributed_graph: # Commented out for the same reason as the multi-masternode stuff below...
            # For now, we are just attempting candidate node offload and then defaulting to User masternode distribution
            elif isinstance(module, nn.ModuleList):
                graph = []
                for layer in module:
                    nodes, subgraph = recurse_model(layer, nodes, candidate_node)
                    graph.append(subgraph)
                return nodes, graph

            else:
                # Spawn secondary DistributedModel on master (subgraph)
                graph_name = f"DistributedModel:{module_name}:MASTER"
                graph = {graph_name: {}}

                for name, submodule in module_children:
                    if len(graph[graph_name]) > 0:
                        candidate_node = max(enumerate(nodes), key=lambda x: x[1]["memory"])[0]

                    # Handle submodule
                    nodes, subgraph = recurse_model(submodule, nodes, candidate_node)

                    graph[graph_name][f"DistributedSubModule:{name}"] = subgraph

                return nodes, graph

        worker_nodes = self.request_workers()
        nodes, graph = recurse_model(model, worker_nodes)



    # def request_worker(self, nodes, module_memory: int, module_type: int):
    #     # module_type = 0 when model is modular , select worker based on lowest latency and having enough memory
    #     if module_type == 0:
    #         candidate_node = max(nodes, key=lambda x: x["memory"])
    #
    #     # module_type = 1 when model is not modular, select the new master based on both the minimum of a latency
    #     # matrix (low latency to other good workers), and memory
    #     elif module_type == 1:
    #         candidate_node = max(nodes, key=lambda x: x["latency"])

    # def acknowledge(self):
    #     pass

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

    def request_workers(self):
        worker_nodes = []
        for i in range(5):  # range(len(self.all_nodes)):
            worker_nodes.append({"id": str(uuid.uuid4()), "memory": 0.5e9})
            # "connection": self.all_nodes[i], "latency_matrix": self.all_nodes[i].latency

        return worker_nodes
