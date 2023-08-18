from src.auth.rsa import generate_rsa_key_pair, load_public_key, authenticate_public_key
from src.p2p.connection import Connection

from cryptography.hazmat.primitives import serialization
from typing import List
import threading
import socket
import time
import random
import ssl
import hashlib
import os


class Node(threading.Thread):

    def __init__(self, host: str, port: int, debug: bool = False, max_connections: int = 0):
        super(Node, self).__init__()

        self.terminate_flag = threading.Event()

        self.host: str = host
        self.port: int = port
        self.id = self.fetch_id() + str(random.random())
        self.debug = debug
        self.max_connections = max_connections

        self.inbound = []
        self.outbound = []
        self.reconnect = []

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.init_sock()

        # self.sock = ssl.wrap_socket(self.sock)

    @property
    def all_nodes(self):
        return self.inbound + self.outbound

    def debug_print(self, message):
        if self.debug:
            print(f"debug: {message}")

    def fetch_id(self):
        cwd = os.getcwd()
        test_dir = os.path.join(cwd, "public_key.pem")

        if not os.path.exists(test_dir):
            generate_rsa_key_pair()

        key = load_public_key(test_dir)

        key_bytes = key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return key_bytes.decode()

    def init_sock(self) -> None:
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(10.0)
        self.sock.listen(1)

    def create_connection(self, connection: socket.socket, id: str, host: str, port: int) -> Connection:
        return Connection(self, connection, id, host, port)

    def send_to_nodes(self, data: bytes, exclude: List[Connection] = None,
                      compression: bool = False) -> None:
        for n in self.all_nodes:
            if exclude is None or n not in exclude:
                self.send_to_node(n, data, compression)

    def send_to_node(self, n: Connection, data: bytes, compression: bool = False) -> None:
        if n in self.all_nodes:
            n.send(data, compression=compression)
        else:
            self.debug_print("send_to_node: node not found!")

    def connect_with_node(self, host: str, port: int, reconnect: bool = False) -> bool:
        if host == self.host and port == self.port:
            self.debug_print(
                "connect_with_node: cannot connect with yourself!")
            return False

        for node in self.all_nodes:
            if node.host == host and node.port == port:
                self.debug_print(
                    f"connect_with_node: already connected with node: {node.id}")
                return True

        node_ids = [node.id for node in self.all_nodes]

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            """
            Area for SSL/TLS implementation and other ID verification + security techniques
            - wrap socket
            - include public rsa key
            - grab host + port via decrypting on-chain info
            """
            self.debug_print(f"connecting to {host} port {port}")
            sock.connect((host, port))

            # ID exchange
            sock.send((self.id + ":" + str(self.port)).encode())
            connected_node_id = sock.recv(4096).decode()

            if authenticate_public_key(connected_node_id) is False:
                sock.send("closing connection: suspicious uid".encode())
                sock.close()
                return False

            # Close process if already connected / self
            if self.id == connected_node_id or connected_node_id in node_ids:
                sock.send("closing connection: already connected".encode())
                sock.close()
                return True

            # Form connection
            thread_client = self.create_connection(
                sock, connected_node_id, host, port)
            thread_client.start()

            self.outbound.append(thread_client)

            # If reconnection to this host is required, it will be added to the list!
            if reconnect:
                self.debug_print(
                    f"connect_with_node: reconnection check enabled on {host}:{port}")
                self.reconnect.append({
                    "host": host, "port": port, "tries": 0
                })

            return True

        except Exception as error:
            self.debug_print(
                f"connect_with_node: could not connect with node. ({error})")
            return False

    def disconnect_with_node(self, node: Connection) -> None:
        if node in self.outbound:
            node.stop()
            self.debug_print(f"node disconnected.")
        else:
            self.debug_print(
                "node disconnect_with_node: cannot disconnect with a node with which we are not connected."
            )

    def stop(self) -> None:
        self.debug_print("node stopping")
        self.terminate_flag.set()

    def reconnect_nodes(self) -> None:
        """This method checks whether nodes that have the reconnection status are still connected. If not
           connected these nodes are started again."""
        for node_to_check in self.reconnect:
            found_node = False
            self.debug_print(
                f"reconnect_nodes: Checking node {node_to_check['host']}:{node_to_check['port']}")

            for node in self.outbound:
                if node.host == node_to_check["host"] and node.port == node_to_check["port"]:
                    found_node = True
                    node_to_check["trials"] = 0  # Reset the trials
                    self.debug_print(
                        f"reconnect_nodes: Node {node_to_check['host']}:{node_to_check['port']} still running!"
                    )

            if not found_node:  # Reconnect with node
                node_to_check["trials"] += 1
                # Perform the actual connection
                self.connect_with_node(
                    node_to_check["host"], node_to_check["port"])

                self.debug_print(
                    f"reconnect_nodes: removing node {node_to_check['host']}:{node_to_check['port']}")
                self.reconnect.remove(node_to_check)

    def run(self):
        while not self.terminate_flag.is_set():
            try:
                connection, client_address = self.sock.accept()

                if self.max_connections == 0 or len(self.inbound) < self.max_connections:

                    # Basic information exchange (not secure) of the id's of the nodes!
                    # backward compatibility
                    connected_node_port = client_address[1]
                    connected_node_id = connection.recv(4096).decode('utf-8')
                    if ":" in connected_node_id:
                        (connected_node_id, connected_node_port) = connected_node_id.split(
                            ':')  # When a node is connected, it sends it id!
                    # Send my id to the connected node!
                    connection.send(self.id.encode('utf-8'))

                    thread_client = self.create_connection(connection, connected_node_id, client_address[0],
                                                           connected_node_port)
                    thread_client.start()

                    self.inbound.append(thread_client)
                    self.outbound.append(thread_client)

                else:
                    self.debug_print(
                        "node: Connection refused: maximum connection limit reached!")
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

    def node_message(self, node: Connection, data):
        time_delta = str(time.time() - float(data))
        self.debug_print(f"node_message: {node.id}: {time_delta}")
