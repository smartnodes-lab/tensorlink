from node_connection import NodeConnection

from typing import List
import threading
import socket


BUFF_SIZE = 4096


class Node(threading.Thread):
    def __init__(self, id: str, host: str, port: int, max_connections: int = 0, debug: bool = False):
        super(Node, self).__init__()

        self.terminate_flag = threading.Event()

        self.id = id
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.debug = debug

        self.nodes = []
        self.message_count = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def init_network(self) -> None:
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(5.0)
        self.sock.listen(1)

    def debug_print(self, message: str) -> None:
        if self.debug:
            print(f"debug {message}")

    def connect_user(self, host: str, port: int, reconnect: bool = False) -> bool:
        if host == self.host and port == self.port:
            self.debug_print(f"Cannot connect to self!")
        elif any(node.host == host and node.port == port for node in self.nodes):
            print("Already connected with node.")
            return True

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.debug_print(f"Connecting to {host}:{port}")
            sock.connect((host, port))

            # Exchange ID info betweeen nodes
            sock.send((self.id + ":" + str(self.port)).encode())
            connected_node_id = sock.recv(BUFF_SIZE).decode()

            if self.id == connected_node_id or connected_node_id in node_ids:
                sock.send("Already connected!".encode())
                sock.close()
                return True

        except Exception as e:
            self.debug_print(f"Could not connect with node: {e}")
            return False

    def send_to_node(self, node: NodeConnection, data: bytes):
        if node in self.nodes:
            node.send(data)
            self.message_count += 1
        else:
            self.debug_print(f"Could not connect to user.")

    def send_to_nodes(self, data: bytes, exclude: List[NodeConnection]):
        nodes = filter(lambda node: node not in exclude, self.nodes)

        for node in nodes:
            self.send_to_node(node, data)
