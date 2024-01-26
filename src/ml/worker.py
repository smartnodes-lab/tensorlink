from src.p2p.smart_node import SmartNode
from src.p2p.connection import Connection
from src.ml.model_analyzer import estimate_memory, get_first_layer, edit_module_code

import torch.nn as nn
import threading
import inspect
import random
import pickle
import torch
import queue
import time
import ast


class DistributedModule(nn.Module):
    def __init__(self, module, master_node, worker_node: Connection):
        super().__init__()
        self.master_node = master_node
        self.worker_node = worker_node

    def forward(self, *args, **kwargs):
        self.master_node.send_tensor((args, kwargs), self.worker_node)
        # Must somehow wait for the response output from the worker

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
        memory += 1.4e9

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
        self.available_memory = get_gpu_memory()

        self.connections = {}
        self.forward_batches = queue.Queue()
        self.backward_batches = queue.Queue()

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
            if b"TENSOR" == data[:6]:
                if self.training:
                    tensor = pickle.loads(data[6:])

                    # Confirm identity/role of node
                    if node in self.inbound:
                        self.forward_batches.put(tensor)
                    elif node in self.outbound:
                        self.backward_batches.put(tensor)

            elif b"MODEL" == data[:5]:
                self.debug_print(f"RECEIVED: {round((data.__sizeof__() - 5) / 1e6, 1)} MB")
                if self.training and not self.model:
                    # Load in model
                    pickled = pickle.loads(data[5:])
                    self.model = pickled
                    self.training = True
                    self.debug_print(f"Loaded submodule: {pickled}")

            elif b"DONE STREAM" == data:
                with open(f"streamed_data_{node.host}_{node.port}", "rb") as f:
                    streamed_bytes = f.read()

                self.stream_data(streamed_bytes, node)

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
            next_node = self.all_nodes[0]
            # Grab backwards pass from forward node and our associated forward pass output
            assoc_forward, backward_batch = self.backward_batches.get()

            # Continue backwards pass on our section of model
            loss = assoc_forward.backward(backward_batch, retain_graph=True)  # Do we need retain graph?
            self.optimizer.zero_grad()
            self.optimizer.step()

            # Pass along backwards pass to next node
            dvalues = get_first_layer(self.model).weight.grad
            self.send_tensor(dvalues.detach(), next_node)

        # Complete any forward pass
        elif self.forward_batches.empty() is False:
            next_node = self.all_nodes[0]
            forward_batch = self.forward_batches.get()
            args, kwargs = forward_batch
            out = self.model(*args, **kwargs)
            self.send_tensor(out.detach(), next_node)

    def send_tensor(self, tensor, node: Connection):
        # tensor_bytes = self.BoT + pickle.dumps(tensor) + self.EoT
        tensor_bytes = b"TENSOR" + pickle.dumps(tensor)
        self.debug_print(f"worker: sending {round(tensor_bytes.__sizeof__() / 1e9, 3)} GB")
        self.send_to_node(node, tensor_bytes)

    def send_module(self, module: nn.Module, node: Connection):
        module_bytes = pickle.dumps(module)
        module_bytes = b"MODEL" + module_bytes
        print("SENDING MODULE")
        self.send_to_node(node, module_bytes)
        time.sleep(10)

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

    def distribute_submodules(self, model: nn.Module):
        """
        Distribute model to connected nodes, assign modules based on memory requirements & latency
        """

        # available_nodes = self.all_nodes
        # candidate_node = max(enumerate([node["memory"] for node in available_nodes]), key=lambda x: x[1])[0]
        candidate_node = self.all_nodes[0]  # Placeholder
        candidate_node_memory = 2e9  # keeps track of offloaded memory to node

        source_code = inspect.getsource(type(model))
        parsed_code = ast.parse(source_code)

        children = dict(model.named_children())

        for node in ast.walk(parsed_code):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                for sub_node in node.body:
                    if isinstance(sub_node, ast.Assign):
                        for target in sub_node.targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.attr in children.keys()
                            ):
                                # Get to submodules defined in main model __init__
                                original_module = getattr(model, target.attr)
                                module_memory = estimate_memory(original_module)  # Must accommodate batch sizes etc

                                # Accommodate on our device if we can
                                if module_memory < self.available_memory:
                                    self.available_memory -= module_memory

                                # Distribute otherwise
                                elif module_memory < candidate_node_memory:
                                    print(f"distributing {target.attr}")
                                    wrapped_module = DistributedModule(original_module, self, candidate_node)
                                    self.send_module(original_module, candidate_node)
                                    setattr(model, target.attr, wrapped_module)
                                    candidate_node_memory -= module_memory

        self.model = model
        
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
