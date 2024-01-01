from src.auth.rsa import get_public_key_obj, authenticate_public_key
from src.p2p.node import Node

from substrateinterface import SubstrateInterface, Keypair
import torch.nn as nn
import torch
import subprocess
import random
import socket
import time


class Worker(Node):
    def __init__(self, host: str, port: int, debug: bool = False, max_connections: int = 0,
                 url: str = "wss://ws.test.azero.dev"):
        super(Worker, self).__init__(host, port, debug, max_connections, url)

        # Smart contract parameters
        self.chain = SubstrateInterface(url=url)
        self.keypair = self.get_substrate_keypair()

        # Hardware parameters


        self.model = None

    def load_model(self, model: nn.Module):
        self.model = model

    def run(self):
        while not self.terminate_flag.is_set():
            try:
                connection, client_address = self.sock.accept()

                if self.max_connections == 0 or len(self.inbound) < self.max_connections:

                    # When a node connects, it sends its public key
                    connected_node_id = connection.recv(4096).decode()

                    # Send random number to confirm their identity
                    randn = str(random.random())
                    message = randn

                    if authenticate_public_key(connected_node_id) is True:
                        id_bytes = connected_node_id.encode()
                        connection.send(
                            self.encrypt(message.encode(), id_bytes)
                        )

                    # Receive a response along with their
                    response = connection.recv(4096).decode()

                    # If node was able to confirm the number, accept identity
                    if response == randn:
                        thread_client = self.create_connection(connection, connected_node_id, client_address[0],
                                                               client_address[1])
                        thread_client.start()

                        self.inbound.append(thread_client)
                        self.outbound.append(thread_client)

                    else:
                        self.debug_print("node: connection refused, invalid ID proof!")
                        connection.close()

                else:
                    self.debug_print(
                        "node: Connection refused: mat reached!")
                    connection.close()

            except socket.timeout:
                self.debug_print('node: Connection timeout!')

            except Exception as e:
                raise e

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
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
            memory = 0

            for device in devices:
                torch.cuda.set_device(device)
                memory_stats = torch.cuda.memory_stats(device)
                device_memory = memory_stats["allocated_bytes.all.peak"] / 1024 / 1024
                memory += device_memory

            print(memory)

    def proof_of_optimization(self):
        pass

    def proof_of_output(self):
        pass

    def proof_of_model(self):
        pass

    def get_substrate_keypair(self):
        # with open("../keys/README.json", "r") as f:
        #     data = json.load(f)
        #
        # encoded_key = base64.b64decode(data["encoded"]).hex()
        # return Keypair.create_from_private_key(private_key=encoded_key, ss58_format=self.chain.ss58_format)
        return Keypair.create_from_uri("//Alice")

