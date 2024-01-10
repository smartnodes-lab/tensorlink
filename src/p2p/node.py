from src.auth.rsa import generate_rsa_key_pair, load_public_key, authenticate_public_key, \
    get_public_key_bytes, load_private_key, get_private_key_bytes, get_public_key_obj
from src.p2p.connection import Connection

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from typing import List
import threading
import base64
import socket
import time


class Node(threading.Thread):
    """
    TODO:
    - add EOF sequence for streaming messages

    """

    def __init__(self, host: str, port: int, debug: bool = False, max_connections: int = 0,
                 url: str = "wss://ws.test.azero.dev"):
        super(Node, self).__init__()

        # User & Connection info
        self.host: str = host
        self.port: int = port
        self.debug = debug
        self.max_connections = max_connections

        # Node & Connection parameters
        self.inbound = []
        self.outbound = []
        self.reconnect = []
        self.terminate_flag = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.init_sock()

        # To add ssl encryption?
        # self.sock = ssl.wrap_socket(self.sock)

    @property
    def all_nodes(self):
        return self.inbound + self.outbound

    def debug_print(self, message):
        if self.debug:
            print(f"{self.host}:{self.port}-debug: {message}")

    def init_sock(self) -> None:
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(3.0)
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
            sock.send(self.get_rsa_pub_key(b=True))
            verification = sock.recv(4096)
            proof = self.decrypt(verification)
            sock.send(proof)

            # # Check if id is a valid rsa public key
            # if authenticate_public_key(node_id) is False:
            #     sock.send("closing connection: suspicious uid".encode())
            #     self.debug_print(f"closing connection: suspicious uid: {node_id}")
            #     sock.close()
            #     return False

            # Add a confirmation mechanism where we send randomized number encrypted via
            # public key to validate identity
            # rand_n = random.random()
            # sock.send(())

            # # Close process if already connected / self (commented out to enable local testing)
            # if self.public_key == connected_node_id or connected_node_id in node_ids:
            #     sock.send("closing connection: already connected".encode())
            #     sock.close()
            #     return True

            # Form connection
            thread_client = self.create_connection(sock, "test-ID", host, port)
            thread_client.start()

            self.outbound.append(thread_client)

            # If reconnection to this host is required, add to the list
            if reconnect:
                self.debug_print(
                    f"connect_with_node: reconnection check enabled on {host}:{port}"
                )
                self.reconnect.append({"host": host, "port": port, "tries": 0})

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
        self.debug_print("Node stopping")
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

    def node_message(self, node: Connection, data):
        time_delta = str(time.time() - float(data))
        self.debug_print(f"node_message: {node.id}: start time: {data}")
        self.debug_print(f"node_message: {node.id}: time delta: {time_delta}")

    def get_rsa_pub_key(self, b=False):
        generate_rsa_key_pair()
        public_key = load_public_key()

        if b is True:
            return get_public_key_bytes(public_key)
        else:
            return public_key

    def get_rsa_priv_key(self, b=False):
        generate_rsa_key_pair()
        private_key = load_private_key()

        if b is True:
            return get_private_key_bytes(private_key)
        else:
            return private_key

    def encrypt(self, data, pub_key: bytes = None):
        # Encrypt the data using RSA-OAEP
        if pub_key is None:
            pub_key = self.get_rsa_pub_key()
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
        private_key = self.get_rsa_priv_key()

        decrypted_data = private_key.decrypt(
            base64.b64decode(data),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return decrypted_data
