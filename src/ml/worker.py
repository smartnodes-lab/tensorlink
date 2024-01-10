from src.p2p.linked_node import LinkedNode

import torch.nn as nn
import torch
import random
import socket
import time
import os


class Worker(LinkedNode):
    """
    TODO:
    - confirm workers public key with smart contract

    """
    def __init__(self, host: str, port: int, debug: bool = False, max_connections: int = 0,
                 url: str = "wss://ws.test.azero.dev"):
        super(Worker, self).__init__(host, port, debug, max_connections, url)

        # Model training parameters
        self.training = False
        self.models = {}

    def run(self):
        while not self.terminate_flag.is_set():
            try:
                connection, client_address = self.sock.accept()

                if self.max_connections == 0 or len(self.inbound) < self.max_connections:
                    self.handshake(connection, client_address)

                else:
                    self.debug_print(
                        "node: Connection refused: Max connections reached!")
                    connection.close()

            except socket.timeout:
                self.debug_print('node: Connection timeout!')

            except Exception as e:
                print(str(e))

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

    def get_gpu_memory(self):
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

    def load_model(self, model: nn.Module):
        self.model = model

    def broadcast_statistics(self):
        memory = str(self.get_gpu_memory())

        if self.training:
            # Incorporate proofs of training
            proof1 = self.proof_of_model()
            proof2 = self.proof_of_optimization()
            proof3 = self.proof_of_output()
        
        self.send_to_nodes(memory.encode())

    def proof_of_optimization(self):
        pass

    def proof_of_output(self):
        pass

    def proof_of_model(self):
        pass

    def get_job(self):
        pass
        # Confirm job details with smart contract, receive initial details from a node?
