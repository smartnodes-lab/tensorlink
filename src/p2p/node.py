import random

from src.p2p.connection import Connection
from src.cryptography.rsa import *

from typing import List
import threading
import socket
import time


class Node(threading.Thread):
    """
    TODO:
    - add EOF sequence for streaming messages
    - ping pong auto-handling
    """

    def __init__(
        self,
        host: str,
        port: int,
        debug: bool = False,
        max_connections: int = 0,
        callback=None,
    ):
        super(Node, self).__init__()

        # User & Connection info
        self.host: str = host
        self.port: int = port
        self.debug = debug
        self.max_connections = max_connections
        self.callback = callback
        self.rsa_pub_key = get_rsa_pub_key(self.port, True)
        self.role = b"W"

        # Node & Connection parameters
        self.connections = []
        self.reconnect = []

        self.terminate_flag = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.init_sock()

        # To add ssl encryption?
        # self.sock = ssl.wrap_socket(self.sock)

    def debug_print(self, message):
        if self.debug:
            print(f"{self.host}:{self.port}-debug: {message}")

    def init_sock(self) -> None:
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(5.0)
        self.sock.listen(1)

    def create_connection(
        self,
        connection: socket.socket,
        host: str,
        port: int,
        node_id=None,
        parent_port: int = None,
        role=None,
    ) -> Connection:
        return Connection(self, connection, host, port, node_id, parent_port, role)

    def send_to_nodes(
        self, data: bytes, exclude: List[Connection] = None, compression: bool = False
    ) -> None:
        for n in self.connections:
            if exclude is None or n not in exclude:
                self.send_to_node(n, data, compression)

    def send_to_node(
        self, n: Connection, data: bytes, compression: bool = False
    ) -> None:
        if n in self.connections:
            n.send(data, compression=compression)
        else:
            self.debug_print("send_to_node: node not found!")

    # def connect_with_node(
    #     self, host: str, port: int, reconnect: bool = False, our_node_id=""
    # ) -> bool:
    #     # Check if trying to connect to self
    #     if host == self.host and port == self.port:
    #         self.debug_print("connect_with_node: cannot connect with yourself!")
    #         return False
    #
    #     # Check if already connected
    #     for node in self.connections:
    #         if node.host == host and node.port == port or node.parent_port == port:
    #             self.debug_print(
    #                 f"connect_with_node: already connected with node: {node.node_id}"
    #             )
    #             return True
    #
    #     # Attempt connection via ID exchange
    #     try:
    #         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         """
    #         Area for SSL/TLS implementation and other ID verification + security techniques
    #         - wrap socket
    #         - include public rsa key
    #         - grab host + port via decrypting on-chain info
    #         """
    #         self.debug_print(f"connecting to {host} port {port}")
    #         # Initialize a socket for potential connection
    #         sock.connect((host, port))
    #
    #         start_time = time.time()
    #
    #         # ID exchange
    #         sock.send(self.role + self.rsa_pub_key)
    #         verification = sock.recv(4096)
    #         latency = time.time() - start_time
    #         verification, their_pub_key = verification.split(b",")
    #         proof = decrypt(verification, self.port)
    #         proof, new_port, node_id = proof.split(b",")
    #
    #         # Now we must verify the other nodes credentials
    #         randn = str(random.random())
    #         message = (
    #             proof
    #             + b","
    #             + f"{self.port}".encode()
    #             + b","
    #             + our_node_id.encode()
    #             + b","
    #             + randn.encode()
    #         )
    #         encrypted_message = encrypt(message, port, their_pub_key)
    #         sock.send(encrypted_message)
    #         response = sock.recv(4096)
    #
    #         if randn == response.decode():
    #             # # Close process if already connected / self (commented out to enable local testing)
    #             # if self.public_key == connected_node_id or connected_node_id in node_ids:
    #             #     sock.send("closing connection: already connected".encode())
    #             #     sock.close()
    #             #     return True
    #
    #             # Form connection
    #             thread_client = self.create_connection(
    #                 sock, host, int(new_port), node_id, port
    #             )
    #             thread_client.start()
    #             thread_client.latency = latency
    #
    #             # self.connections.append(thread_client)
    #             self.connections.append(thread_client)
    #
    #             # If reconnection to this host is required, add to the list
    #             if reconnect:
    #                 self.debug_print(
    #                     f"connect_with_node: reconnection check enabled on {host}:{port}"
    #                 )
    #                 self.reconnect.append({"host": host, "port": port, "tries": 0})
    #
    #             return True
    #
    #     except Exception as error:
    #         self.debug_print(
    #             f"connect_with_node: could not connect with node. ({error})"
    #         )
    #         return False

    def disconnect_with_node(self, node: Connection) -> None:
        if node in self.connections:
            node.stop()
            self.debug_print(f"node disconnected.")
        else:
            self.debug_print(
                "node disconnect_with_node: cannot disconnect with a node with which we are not connected."
            )

    def stop(self) -> None:
        self.debug_print("Node stopping")
        self.terminate_flag.set()

    def reconnect_nodes(self) -> None:
        """This method checks whether nodes that have the reconnection status are still connected. If not
        connected these nodes are started again."""
        for node_to_check in self.reconnect:
            found_node = False
            self.debug_print(
                f"reconnect_nodes: Checking node {node_to_check['host']}:{node_to_check['port']}"
            )

            for node in self.connections:
                if (
                    node.host == node_to_check["host"]
                    and node.port == node_to_check["port"]
                ):
                    found_node = True
                    node_to_check["trials"] = 0  # Reset the trials
                    self.debug_print(
                        f"reconnect_nodes: Node {node_to_check['host']}:{node_to_check['port']} still running!"
                    )

            if not found_node:  # Reconnect with node
                node_to_check["trials"] += 1
                # Perform the actual connection
                self.connect_with_node(node_to_check["host"], node_to_check["port"])

                self.debug_print(
                    f"reconnect_nodes: removing node {node_to_check['host']}:{node_to_check['port']}"
                )
                self.reconnect.remove(node_to_check)

    def handle_message(self, node: Connection, data):
        self.debug_print(
            f"handle_message from {node.host}:{node.port} -> {data.__sizeof__()/1e6}MB"
        )

        if self.callback is not None:
            self.callback(data, node)

    # def measure_latency(self, node: Connection):
    #     start_time = time.time()
