from src.auth.rsa import get_public_key_obj
from src.p2p.node import Node

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
import random
import base64
import socket
import time


class Worker(Node):
    def __init__(self, host: str, port: int, debug: bool = False, max_connections: int = 0,
                 url: str = "wss://ws.test.azero.dev"):
        super(Worker, self).__init__(host, port, debug, max_connections, url)

    def run(self):
        while not self.terminate_flag.is_set():
            try:
                connection, client_address = self.sock.accept()

                if self.max_connections == 0 or len(self.inbound) < self.max_connections:

                    # When a node connects, it sends its public key
                    connected_node_id = connection.recv(4096).decode()

                    # Send random number to confirm their identity\
                    randn = str(random.random())

                    connection.send(
                        self.encrypt(randn, connected_node_id)
                    )

                    [response, node_id, node_port] = connection.recv(4096).decode().split(",")

                    if response == randn:
                        thread_client = self.create_connection(connection, connected_node_id, client_address[0],
                                                               connected_node_port)
                    thread_client.start()

                    self.inbound.append(thread_client)
                    self.outbound.append(thread_client)

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

    def encrypt(self, data, pub_key: bytes = None):
        # Encrypt the data using RSA-OAEP
        if pub_key is None:
            pub_key = self.get_public_key()
        else:
            pub_key = get_public_key_obj(pub_key)

        encrypted_data = pub_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return base64.b64encode(encrypted_data)

    def decrypt(self, data):
        private_key = self.get_private_key()

        decrypted_data = private_key.decrypt(
            base64.b64decode(data),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return decrypted_data

    def proof_of_optimization(self):
        pass

    def proof_of_output(self):
        pass

    def proof_of_model(self):
        pass
