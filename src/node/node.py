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

        self.nodes_connected = []
        self.nodes_lost = []

        self.message_count = 0
        self.reconnect_attempts = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def init_network(self) -> None:
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(5.0)
        self.sock.listen(1)

    def debug_print(self, message: str) -> None:
        if self.debug:
            print(f"debug {message}")

    def print_connections(self) -> None:
        print("Node connection overview:")
        print(f"Total nodes connected: {len(self.nodes_connected)}")

    def connect_node(self, host: str, port: int, reconnect: bool = False) -> bool:
        if host == self.host and port == self.port:
            self.debug_print(f"Cannot connect to self!")
        elif any(node.host == host and node.port == port for node in self.nodes_connected):
            print("Already connected with node.")
            return True

        node_ids = [node.id for node in self.nodes_connected]

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.debug_print(f"Connecting to {host}:{port}")
            sock.connect((host, port))

            # Exchange ID info between nodes
            sock.send((self.id + ":" + str(self.port)).encode())
            connected_node_id = sock.recv(BUFF_SIZE).decode()

            if self.id == connected_node_id or connected_node_id in node_ids:
                sock.send("Already connected to client!".encode())
                sock.close()
                return True

            thread_client = self.create_connection(sock, connected_node_id, host, port)
            thread_client.start()

            self.nodes_connected.append(thread_client)
            self.debug_print(f"Node connected!")

            if reconnect:
                self.debug_print(f"Reconnection ")
                self.nodes_lost.append({
                    "host": host, "port": port, "tries": 0
                })

            return True

        except Exception as e:
            self.debug_print(f"Could not connect with node: {e}")
            return False

    def create_connection(self, connection: socket.socket, id: str, host: str, port: int) -> NodeConnection:
        return NodeConnection(self, connection, id, host, port)

    def delete_connection(self, node: NodeConnection) -> None:
        if node in self.nodes_connected:
            self.debug_print(f"Disconnecting with node: {node.host}:{self.port}")
            node.stop()
        else:
            self.debug_print(f"Cannot disconnect with node {node.host}:{node.port} (does not exist!)")

    def stop(self) -> None:
        self.debug_print(f"Node requested to stop.")
        self.terminate_flag.set()

    def send_to_node(self, node: NodeConnection, data: bytes) -> None:
        if node in self.nodes_connected:
            node.send(data)
            self.message_count += 1
        else:
            self.debug_print(f"Could not connect to user.")

    def send_to_nodes(self, data: bytes, exclude: List[NodeConnection]) -> None:
        nodes = filter(lambda node: node not in exclude, self.nodes_connected)

        for node in nodes:
            self.send_to_node(node, data)

    def reconnect_nodes(self) -> None:
        for lost_node in self.nodes_lost:
            lh = lost_node.host
            lp = lost_node.port

            found = False
            self.debug_print(f"Attempting to reconnect with {lh}:{lp}")

            for node in self.nodes_connected:
                if node.host == lh and node.port == lp:
                    found = True
                    lost_node.reconnect_attempts = 0
                    self.debug_print(f"Reconnected to node {node.host}:{node.port}")

            if not found:
                lost_node.reconnect_attempts += 1
                self.debug_print(f"Reconnecting to node {lh}:{lp} "
                                 f"(Attempt {lost_node.reconnect_attempts})")
                self.connect_node(lh, lp)

    def run(self) -> None:
        while not self.terminate_flag.is_set():
            try:
                self.debug_print("Awaiting connections.")
                # self.model.train_loop(1) from previous rendition of neuronn
                connection, client_address = self.sock.accept()

                self.debug_print(f"Total connections: " + str(self.nodes_connected))

                if self.max_connections == 0 or len(self.nodes_connected) < self.max_connections:
                    # Non-secured node-id exchange
                    client_address = [1]

            except:
                pass

    def node_message(self, node: NodeConnection, data):
        self.debug_print(f"node-message::{node.id}: {data}")

    def node_disconnect(self, node: NodeConnection):
        if node in self.nodes_connected:
            self.debug_print(f"node::disconnecting::{node.id}")
            del self.nodes_connected[self.nodes_connected.index(node)]
