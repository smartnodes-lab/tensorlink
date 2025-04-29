from tensorlink.crypto.rsa import (
    decrypt,
    encrypt,
    authenticate_public_key,
    get_rsa_pub_key,
)
from tensorlink.p2p.connection import Connection
from tensorlink.p2p.monitor import ConnectionMonitor

from logging.handlers import TimedRotatingFileHandler
from dotenv import get_key, set_key
from typing import Tuple, Union
from miniupnpc import UPnP
from web3 import Web3
import hashlib
import ipaddress
import json
import logging
import os
import random
import requests
import socket
import threading
import time


COLOURS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}


# Grab smart contract information
base_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(base_dir, "../config")
SM_CONFIG_PATH = os.path.join(CONFIG_PATH, "SmartnodesCore.json")
MS_CONFIG_PATH = os.path.join(CONFIG_PATH, "SmartnodesMultiSig.json")
API = get_key(".tensorlink.env", "API")

with open(os.path.join(CONFIG_PATH, "config.json"), "r") as f:
    config = json.load(f)
    CHAIN_URL = config["api"]["chain-url"]
    if API:
        CHAIN_URL = API

    CONTRACT = config["api"]["core"]
    MULTI_SIG_CONTRACT = config["api"]["multi-sig"]

with open(SM_CONFIG_PATH, "r") as f:
    METADATA = json.load(f)
ABI = METADATA["abi"]

with open(MS_CONFIG_PATH, "r") as f:
    MS_METADATA = json.load(f)
MULTI_SIG_ABI = MS_METADATA["abi"]

SNO_EVENT_SIGNATURES = {
    "JobRequest": "JobRequested(uint256,uint256,address[])",
    "JobComplete": "JobCompleted(uint256,uint256)",
    "JobDispute": "JobDisputed(uint256,uint256)",
    "ProposalCreated": "ProposalCreated(uint256,bytes)",
    "ProposalExecuted": "ProposalExecuted(uint256)",
}


# Configure logging with TimedRotatingFileHandler
os.makedirs("logs", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

log_handler = TimedRotatingFileHandler(
    "logs/runtime.log", when="midnight", interval=1, backupCount=30
)
log_handler.setFormatter(logging.Formatter("[%(asctime)s] - %(message)s"))
log_handler.suffix = "%Y%m%d"
logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.DEBUG)
BASE_PORT = 38751


def hash_key(key: bytes, number=False):
    """
    Hashes the key to determine its position in the keyspace.
    """
    if number is True:
        return int(hashlib.sha256(key).hexdigest(), 16)
    else:
        return hashlib.sha256(key).hexdigest()


def calculate_xor(key_hash, node_id):
    """
    Calculate the XOR distance between a key and a nodes ID.
    """
    return int(key_hash, 16) ^ int(node_id, 16)


def get_public_ip():
    """Get the public IP address of the local machine."""
    try:
        response = requests.get("https://api.ipify.org")
        return response.text
    except requests.RequestException as e:
        print(f"Error retrieving public IP: {e}")
        return None


def is_private_ip(ip):
    """Check if the IP address is private."""
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def get_connection_info(node, main_port=None, upnp=True):
    """Connection info for routing table storage"""
    node_host = node.host

    if is_private_ip(node_host) and upnp:
        node_host = get_public_ip()

    info = {
        "host": node_host,
        "port": node.port if not main_port else main_port,
        "role": node.role,
        "id": node.node_id,
        "reputation": node.reputation,
    }

    return info


def log_entry(node, metadata):
    entry = {"timestamp": time.ctime(), "node": node.node_id, "metadata": metadata}
    logging.info(json.dumps(entry))


def clean():
    for file in os.listdir(os.path.join(os.getcwd(), "tmp")):
        if "streamed_data" in file.title():
            path = os.path.join(os.path.join(os.getcwd(), "tmp"), file)
            os.remove(path)


def _parse_initial_connection(connection: socket.socket) -> dict:
    """Parse and extract initial connection information"""
    id_info = connection.recv(1024)
    _, role, node_address, connected_node_id = json.loads(id_info)
    connected_node_id = connected_node_id.encode()
    node_id_hash = hashlib.sha256(connected_node_id).hexdigest()

    return {
        'role': role,
        'node_address': node_address,
        'node_id': connected_node_id,
        'node_id_hash': node_id_hash,
        'random': _,
    }


class Bucket:
    """A bucket for storing local values in the Kademlia-inspired DHT"""

    def __init__(self, distance_from_key, max_values):
        self.values = []
        self.distance_from_key = distance_from_key
        self.max_values = max_values * (2**distance_from_key)

    def is_full(self):
        return len(self.values) >= self.max_values

    def add_node(self, value):
        if not self.is_full():
            self.values.append(value)

    def remove_node(self, value):
        if value in self.values:
            self.values.remove(value)


class SmartNode(threading.Thread):
    """
    A P2P nodes secured by RSA encryption and smart contract validation for the Smartnodes ecosystem.
    Combines smart contract queries and a kademlia-like DHT implementation for data storage and access.
    """

    def __init__(
        self,
        role,
        max_connections: int = 0,
        upnp: bool = True,
        off_chain_test: bool = False,
        local_test: bool = False,
        debug_colour=None,
    ):
        super(SmartNode, self).__init__()

        # Node info
        self.terminate_flag = threading.Event()
        self.connection_listener = None

        # Connection info
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clean()

        # Get private ip
        if not local_test:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.host = s.getsockname()[0]
            s.close()
        else:
            self.host = "127.0.0.1"

        self.port = BASE_PORT
        self.used_ports = set()
        self.max_connections = max_connections
        self.print_level = logging.INFO

        # Connection Settings
        self.upnp = upnp
        self.nodes = {}  # node hash: Connection
        self.max_attempts_per_minute = 5
        self.block_duration = 600
        self.rate_limiter = ConnectionMonitor(
            self.max_attempts_per_minute, self.block_duration
        )

        self.debug_colour = None
        if debug_colour:
            self.debug_colour = debug_colour

        # DHT Parameters
        self.replication_factor = 3
        self.bucket_size = 2
        self.buckets = [Bucket(d, self.bucket_size) for d in range(256)]
        self.routing_table = {}
        self.requests = {}

        # More parameters for smart contract / p2p info
        self.role = role
        self.rsa_pub_key = get_rsa_pub_key(self.role, True)
        self.rsa_key_hash = hashlib.sha256(self.rsa_pub_key).hexdigest()

        # Stores key of stored values
        self.validators = []
        self.workers = []
        self.users = []
        self.jobs = []

        self.sno_events = {
            name: Web3.keccak(text=sig).hex()
            for name, sig in SNO_EVENT_SIGNATURES.items()
        }
        self.off_chain_test = off_chain_test
        self.local_test = local_test

        if local_test:
            self.upnp = False
            self.off_chain_test = True

        self.public_key = None

        if self.upnp:
            self._init_upnp()

        self._init_sock()

        if self.off_chain_test is False:
            # Smart nodes parameters for additional security and contract connectivity
            self.url = CHAIN_URL
            self.chain = Web3(Web3.HTTPProvider(CHAIN_URL))
            self.contract_address = Web3.to_checksum_address(CONTRACT)

            # Grab the SmartNode contract
            try:
                self.contract = self.chain.eth.contract(
                    address=self.contract_address, abi=ABI
                )
                self.multi_sig_contract = self.chain.eth.contract(
                    address=MULTI_SIG_CONTRACT, abi=MULTI_SIG_ABI
                )

            except Exception as e:
                self.debug_print(
                    f"SmartNode -> Could not retrieve contract: {e}",
                    colour="bright_red",
                    level=logging.CRITICAL,
                )
                self.stop()

    def handle_data(self, data: bytes, node: Connection) -> bool:
        """
        Process incoming data from node connections based on message type.

        Args:
            data (bytes): Incoming data packet
            node (Connection): Source node connection

        Returns:
            bool: True if data was processed, False if message type unknown
        """
        try:
            # Handle streamed data completion
            if data.startswith(b"DONE STREAM"):
                return self._handle_stream_completion(data, node)

            # Handle ping/pong communication
            if data.startswith(b"PING"):
                self.send_to_node(node, b"PONG")
                return True

            if data.startswith(b"PONG"):
                return self._handle_pong_response(node)

            # Handle distributed hash table (DHT) value requests and responses
            if data.startswith(b"REQUEST-VALUE-RESPONSE"):
                return self._handle_value_response(data, node)

            if data.startswith(b"REQUEST-VALUE"):
                return self._handle_value_request(data, node)

            # Unknown message type
            return False

        except Exception as e:
            self._log_error(f"Error handling data: {e}")
            return False

    def _handle_stream_completion(self, data: bytes, node: Connection) -> bool:
        """
        Process completed data streams from nodes.

        Args:
            data (bytes): Completed stream data
            node (Connection): Source node connection

        Returns:
            bool: True if stream processed successfully
        """
        file_name = f"tmp/streamed_data_{node.host}_{node.port}_{self.host}_{self.port}"

        # Handle module or parameter stream
        if b"MODULE" in data[11:]:
            os.rename(file_name, data[17:].decode())
            streamed_bytes = data[11:]
        elif b"PARAMETERS" in data[11:]:
            os.rename(file_name, f"tmp/{data[21:].decode()}_parameters")
            streamed_bytes = data[11:]
        else:
            # Read and remove temporary file for generic streams
            with open(file_name, "rb") as file:
                streamed_bytes = file.read()

            os.remove(file_name)

        # Recursively process streamed data
        self.handle_data(streamed_bytes, node)

        return True

    def _handle_pong_response(self, node: Connection) -> bool:
        """
        Update node latency based on ping-pong timing.

        Args:
            node (Connection): Node that sent pong response

        Returns:
            bool: True if pong processed successfully
        """
        if node.pinged > 0:
            node.ping = time.time() - node.pinged
            node.pinged = -1
        else:
            self.debug_print("Received pong with no corresponding ping", colour="red")
            node.ghosts += 1

        return True

    def _handle_value_response(self, data: bytes, node: Connection) -> bool:
        """
        Process value response from a DHT node.

        Args:
            data (bytes): Response data packet
            node (Connection): Source node connection

        Returns:
            bool: True if response processed successfully
        """
        # Validate response packet size
        if len(data) < 86:
            self.debug_print("Received incomplete value response", colour="red")
            return False

        # Check if we have an active request for this node
        if node.node_id not in self.requests:
            self.debug_print("Received unsolicited data", colour="red")
            return False

        value_id = data[22:86].decode()
        request_key = f"REQUEST-VALUE{value_id}"

        # Process only if this is a known request
        if request_key in self.requests[node.node_id]:
            value = json.loads(data[86:])
            self.requests[node.node_id].remove(request_key)
            self._store_request(value_id, value)
            return True

        self.debug_print(f"Ghost data from node: {node.node_id}", colour="red")
        node.ghosts += 1
        return False

    def _handle_value_request(self, data: bytes, node: Connection) -> bool:
        """
        Respond to a value request from another node.

        Args:
            data (bytes): Request data packet
            node (Connection): Requesting node connection

        Returns:
            bool: True if request processed successfully
        """
        # Validate request packet size
        if len(data) < 141:
            return False

        value_hash = data[13:77].decode()
        requester = data[77:141].decode()

        # Query local DHT for requested value
        value = self.query_dht(value_hash, requester)

        # Send response back to requesting node
        response = (
            b"REQUEST-VALUE-RESPONSE" + value_hash.encode() + json.dumps(value).encode()
        )
        self.send_to_node(node, response)

        return True

    def _log_error(self, message: str) -> None:
        """
        Log error messages with appropriate severity.

        Args:
            message (str): Error message to log
        """
        self.debug_print(
            f"SmartNode -> {message}", colour="bright_red", level=logging.ERROR
        )

    def debug_print(self, message, level=logging.DEBUG, colour=None) -> None:
        """Print to console if debug is enabled"""
        logging.log(level, message)

        if level >= self.print_level:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            role_colour = "\033[37m"
            if self.role == "U":
                role_colour = COLOURS["magenta"]
            elif self.role == "W":
                role_colour = COLOURS["cyan"]
            elif self.role == "V":
                role_colour = COLOURS["yellow"]

            if colour is None or colour not in COLOURS.keys():
                colour = "\033[37m"
            else:
                colour = COLOURS[colour]

            reset_colour = "\033[0m"
            print(f"{role_colour}{timestamp}{colour} -> {message}{reset_colour}")

    """Methods for DHT Query and Storage"""

    def query_dht(self, key_hash, requester: str = None, ids_to_exclude: list = None):
        """
        Retrieve stored value from DHT or query the closest nodes to a given key.
        * should be run in its own thread due to blocking RPC
        """
        if isinstance(key_hash, bytes):
            key_hash = key_hash.decode()

        self.debug_print(f"SmartNode -> Querying DHT for {key_hash}")
        closest_node = None
        closest_distance = float("inf")

        # Find nearest nodes in our routing table
        for node_hash, node in self.routing_table.items():
            # Get XOR distance between keys
            distance = calculate_xor(key_hash.encode(), node_hash.encode())

            if distance < closest_distance:
                if not ids_to_exclude or node_hash not in ids_to_exclude:
                    closest_node = (node_hash, node)
                    closest_distance = distance

        if requester is None:
            requester = self.rsa_key_hash

        if closest_node is not None:
            # The case where the stored value was not properly deleted (ie is None)
            if closest_node[1] is None:
                self._delete_item(closest_node[0])

                # Get another closest nodes
                return self.query_dht(key_hash, requester, ids_to_exclude)

            # The case where we have the stored value
            elif closest_node[0] == key_hash:
                # The value is a stored data structure and we can return it
                return closest_node[1]

            # We don't have the stored value, and must route the request to the nearest nodes
            else:
                if closest_node[0] in self.validators:
                    closest_node_hash = closest_node[1]["id"]
                    return self.query_node(
                        key_hash,
                        self.nodes[closest_node_hash],
                        requester,
                        ids_to_exclude,
                    )

        else:
            return None

    def query_node(
        self,
        key_hash: Union[str, bytes],
        node: Connection,
        requester: Union[str, bytes] = None,
        ids_to_exclude: list = None,
    ):
        """Query a specific nodes for a value"""
        if requester is None:
            requester = self.rsa_key_hash

        if node.terminate_flag.is_set():
            return

        if isinstance(key_hash, bytes):
            key_hash = key_hash.decode()
        if isinstance(requester, bytes):
            requester = requester.decode()

        start_time = time.time()
        message = b"REQUEST-VALUE" + key_hash.encode() + requester.encode()
        self.send_to_node(node, message)

        # Logs what value were requesting and from what nodes
        self._store_request(node.node_id, "REQUEST-VALUE" + key_hash)

        # Blocking wait to receive the data
        while "REQUEST-VALUE" + key_hash in self.requests[node.node_id]:
            # TODO some better timeout management
            # Wait for 3 seconds and then find a new nodes to query
            if time.time() - start_time > 3:
                if ids_to_exclude is not None:
                    if len(ids_to_exclude) > 1:
                        return None
                    ids_to_exclude.append(node.node_id)
                else:
                    ids_to_exclude = [node.node_id]

                # Re route request to the next closest nodes
                self.requests[node.node_id].remove(key_hash)
                return self.query_dht(key_hash, requester, ids_to_exclude)

            if ids_to_exclude and len(ids_to_exclude) > 1:
                return None

        return_val = self.requests[key_hash][-1]

        return return_val

    def store_value(self, key: str, value: object, replicate: object = 0) -> object:
        """Store value in routing table and replicate if specified"""
        bucket_index = self.calculate_bucket_index(key)
        bucket = self.buckets[bucket_index]

        if not bucket.is_full() or key in bucket:
            self.routing_table[key] = value
            bucket.add_node(key)

        if 5 > replicate > 0:
            # TODO
            # n_validators = self.get_validator_count()

            while replicate > 0:
                # random_id = random.randrange(1, n_validators + 1)

                replicate -= 1
                pass

    def request_store_value(self):
        pass

    def _store_request(self, node_id: str, key: str):
        """Stores a log of the request we have made to a nodes and for what value"""
        if node_id in self.requests:
            self.requests[node_id].append(key)
        else:
            self.requests[node_id] = [key]

    def _remove_request(self, node_id: str, key: str):
        if node_id in self.requests:
            self.requests[node_id].remove(key)

    def _delete_item(self, key: str):
        """
        Delete a key-value pair from the DHT.
        """

        if key in self.routing_table:
            bucket_index = self.calculate_bucket_index(key)
            bucket = self.buckets[bucket_index]
            bucket.remove_node(key)

            if key in self.nodes or self.jobs:
                # Do not delete information related to active connections or jobs
                return

            del self.routing_table[key]
            self.debug_print(f"SmartNode -> Key {key} deleted from DHT.", colour="blue")
        else:
            self.debug_print(
                f"SmartNode -> Key {key} not found in DHT.",
                colour="red",
                level=logging.ERROR,
            )

    def calculate_bucket_index(self, key: str):
        """
        Find the index of a bucket given the key
        """
        key_int = int(key.encode(), 16)
        bucket_index = key_int % len(self.buckets)

        return bucket_index

    """Peer-to-peer methods"""

    def _listen(self):
        """Listen for incoming connections and initialize custom handshake"""
        while not self.terminate_flag.is_set():
            try:
                if self.sock.fileno() == -1:
                    return

                self.sock.settimeout(2.0)

                try:
                    # Unpack nodes info
                    connection, node_address = self.sock.accept()
                    ip_address = node_address[0]
                except socket.timeout:
                    continue

                # Check rate limiting
                if self.rate_limiter.is_blocked(ip_address):
                    self.close_connection(
                        connection,
                        f"listen: connection refused, rate limit exceeded for {ip_address}",
                    )
                    continue

                self.rate_limiter.record_attempt(ip_address)

                # Attempt custom handshake
                if self.max_connections == 0 or len(self.nodes) < self.max_connections:
                    self._handshake(connection, node_address)

                else:
                    self.close_connection(
                        connection,
                        "listen: connection refused, max connections reached",
                    )

            except socket.timeout:
                # self.debug_print(f"listen: connection timed out!")
                pass

            except Exception as e:
                self.debug_print(
                    f"SmartNode -> listen connection error {e}",
                    colour="bright_red",
                    level=logging.CRITICAL,
                )

            # self.reconnect_nodes()

    def _handshake(
        self, connection: socket.socket, node_address, instigator=False
    ) -> bool:
        """
        Orchestrate the handshake process with a new node. Validates incoming node's keys via a random
        number swap, along with SC verification of connecting user.

        This handshake method is called by both the instigator (connecting node) and receiving node. Both first
        verify on-chain credentials of the other node and determine if they should connect. After this, the
        connection instigator decrypts a random number generated by the remote node with their private key to
        confirm their identity. This is then reciprocated on the other node.
        """
        try:
            connection.settimeout(10)

            # Receive and parse initial node information
            node_info = _parse_initial_connection(connection)

            # Perform reputation check and role-specific checks
            if not self._validate_node_credentials(node_info, connection):
                return False

            # Verify node's identity via cryptographic proofs
            verified, our_port, new_port, main_port = self._verify_node_identity(
                connection, node_info, instigator, node_address
            )

            if not verified:
                return False

            # Establish the connection
            return self._finalize_connection(
                our_port, new_port, main_port, instigator, node_info, node_address
            )

        except Exception as e:
            self.close_connection(connection, f"Handshake failed: {e}")
            return False

    def _validate_node_credentials(
        self, node_info: dict, connection: socket.socket
    ) -> bool:
        """Perform comprehensive credential validation"""

        # Check node reputation from validator nodes
        if len(self.nodes) > 0 and self.off_chain_test is False:
            dht_info = self.query_dht(
                node_info['node_id_hash'], ids_to_exclude=[node_info['node_id_hash']]
            )

            if dht_info and dht_info.get("reputation", 0) < 40:
                self.debug_print(
                    f"Poor reputation: {node_info['node_id_hash']}",
                    colour="red",
                    level=logging.WARNING,
                )
                connection.close()
                return False

        # If we are a worker, only allow connections from users which we have been coordinated to work with
        # by a validator (i.e. node_id_hash is in self.requests)
        if node_info['role'] == "U" and self.role == "W":
            if node_info['node_id_hash'] not in self.requests:
                self.close_connection(
                    connection, f"Not assigned to user: {node_info['node_id_hash']}"
                )
                return False

        # Public key validation
        if not authenticate_public_key(node_info['node_id']):
            self.close_connection(connection, "Invalid RSA key")
            return False

        # On-chain validation
        return self._validate_on_chain_credentials(node_info, connection)

    def _validate_on_chain_credentials(
        self, node_info: dict, connection: socket.socket
    ) -> bool:
        """Validate node credentials against on-chain information"""
        if self.off_chain_test:
            return True

        try:
            if node_info['role'] == "V":
                is_active, pub_key_hash = self.contract.functions.getValidatorInfo(
                    node_info['node_address']
                ).call()

                if not is_active or node_info['node_id_hash'] != bytes.hex(
                    pub_key_hash
                ):
                    self.close_connection(
                        connection, f"Validator not listed: {node_info['node_id']}"
                    )
                    return False

            elif node_info['role'] not in ["W", "U"]:
                self.close_connection(connection, f"Invalid role: {node_info['role']}")
                return False

        except Exception as e:
            self.close_connection(connection, f"Contract query error: {e}")
            return False

        return True

    def _verify_node_identity(
        self,
        connection: socket.socket,
        node_info: dict,
        instigator: bool,
        node_address: list,
    ) -> Tuple[bool, int, int, int]:
        """Perform cryptographic identity verification"""
        rand_n = random.random()
        encrypted_number = encrypt(
            str(rand_n).encode(), self.role, node_info['node_id']
        )

        # Prepare verification message
        if instigator:
            proof = self._handle_instigator_proof(node_info)
            if proof is None:
                return False

            message = json.dumps((self.port, proof, encrypted_number.decode()))

        else:
            message = json.dumps(
                (
                    encrypted_number.decode(),
                    self.role,
                    self.public_key,
                    self.rsa_pub_key.decode(),
                )
            )

        connection.send(message.encode())

        # Verify response
        return self._validate_response(connection, rand_n, instigator, node_address)

    def _handle_instigator_proof(self, node_info: dict) -> float:
        """Handle proof verification for connection instigator"""
        try:
            proof = decrypt(node_info['random'].encode(), self.role)
            return float(proof)
        except Exception as e:
            self.debug_print(f"Proof validation failed: {e}")
            return 0

    def _validate_response(
        self,
        connection: socket.socket,
        expected_rand_n: float,
        instigator: bool,
        node_address: list,
    ) -> Tuple[bool, int, int, int]:
        """
        Validate the response from the remote node. Perform a random number swap, and once confirmed swap
        non-instigator node to a new port.
        """
        try:
            response = connection.recv(1024)

            if instigator:
                main_port = node_address[1]
                new_port, rand_n_proof = json.loads(response)
                rand_n_proof = float(rand_n_proof)
                our_port = connection.getsockname()[1]
                connection.close()
            else:
                # Unpack data from node
                response = json.loads(response)
                main_port, rand_n_proof, verification = response
                verification = decrypt(verification.encode(), self.role)

                # Select a new port for the node to use (since we accepted connection from the listening/main socket)
                our_port = self._get_next_port()
                self.debug_print(
                    f"SmartNode -> Selected next port: {our_port} for new connection.",
                    level=logging.DEBUG,
                )
                self.add_port_mapping(our_port, our_port)

                # Send the new port and proof of random number
                response = json.dumps((our_port, verification.decode()))
                new_port = node_address[1]
                connection.send(response.encode())
                connection.close()

            return rand_n_proof == expected_rand_n, our_port, new_port, main_port

        except Exception as e:
            self.debug_print(f"Response validation failed: {e}")
            return False, 0, 0, 0

    def _finalize_connection(
        self,
        our_port: int,
        new_port: int,
        main_port: int,
        instigator: bool,
        node_info: dict,
        node_address: tuple,
    ) -> bool:
        """Finalize the node connection process"""
        if instigator:
            try:
                self.debug_print(
                    f"Switching connection to new port: {node_address[0]}:{new_port}"
                )
                time.sleep(1)
                new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                new_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                new_sock.connect((node_address[0], new_port))
                new_sock.settimeout(6)
                connection = new_sock
            except Exception as e:
                self.debug_print(f"Port swap failed ({self.host}:{our_port}): {e}")
                raise e
        else:
            self.debug_print(
                f"SmartNode -> Listening for the instigator on the new port: {our_port}"
            )
            time.sleep(1)

            # Create a new socket and bind to the selected port
            new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            new_sock.bind((self.host, our_port))
            new_sock.settimeout(6)
            new_sock.listen(5)

            # Accept the incoming connection on the new port
            connection, new_node_address = new_sock.accept()

        # Check for existing connection
        for node in self.nodes.values():
            if node.host == node_address[0] and node.port == node_address[1]:
                self.debug_print(f"Already connected to node: {node.node_id}")
                connection.close()
                return False

        # Create connection thread
        thread_client = self._create_connection(
            connection,
            host=node_address[0],
            port=new_port,
            main_port=main_port,
            node_id=node_info['node_id'],
            role=node_info['role'],
        )
        thread_client.start()

        # Store and categorize node
        return self._store_node_connection(thread_client, node_info)

    def _store_node_connection(self, thread_client, node_info: dict) -> bool:
        """Store node connection details and categorize"""
        if thread_client.terminate_flag.is_set():
            return False

        # Retrieve or generate node information
        stored_info = self.query_dht(node_info['node_id_hash']) or get_connection_info(
            thread_client,
            main_port=thread_client.main_port,
            upnp=False if self.upnp is None else True,
        )

        # Store node details
        self.nodes[node_info['node_id_hash']] = thread_client
        self.store_value(node_info['node_id_hash'], stored_info)

        # Categorize node by role
        if node_info["role"] == "V":
            self.validators.append(node_info["node_id_hash"])

        elif node_info["role"] == "W":
            self.workers.append(node_info["node_id_hash"])

            # If we are connecting to a worker (for a job), boost connection speed
            if self.role == "U":
                thread_client.adjust_chunk_size("large")

        elif node_info["role"] == "U":
            self.users.append(node_info["node_id_hash"])

            # If we are connecting to a user (for a job), boost connection speed
            if self.role == "W":
                thread_client.adjust_chunk_size("large")

        self.debug_print(
            f"Connected to node: {thread_client.host}:{thread_client.port}",
            level=logging.INFO,
            colour="bright_green",
        )

        return True

    def _handle_user_connection(self, thread_client, node_info: dict):
        """Special handling for user connections"""
        self.users.append(node_info['node_id_hash'])
        if self.role == "W":
            thread_client.adjust_chunk_size("large")

    def connect_node(
        self, id_hash: Union[bytes, str], host: str, port: int, reconnect: bool = False
    ) -> bool:
        """
        Connect to a node and exchange information to confirm its role in the Smartnodes network.
        """
        if isinstance(id_hash, bytes):
            id_hash = id_hash.decode()

        _can_connect = self._can_connect(host, port)

        # Check that we are not already connected
        if id_hash in self.nodes:
            self.debug_print(
                f"SmartNode -> connect_node: Already connected to {id_hash}"
            )
            return True

        if _can_connect:
            backoff = 1
            max_attempts = 3

            for attempt in range(max_attempts):
                try:
                    # Open up a free port
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    our_port = self._get_next_port()
                    self.add_port_mapping(our_port, our_port)
                    self.debug_print(
                        f"SmartNode -> Selected next port: {our_port} for new connection"
                    )

                    if self.local_test:
                        host = "127.0.0.1"

                    # Attempt connection
                    sock.bind((self.host, our_port))
                    sock.connect((host, port))
                    sock.settimeout(10)

                    self.debug_print(
                        f"SmartNode -> connect_node: connecting to {host}:{port}",
                        colour="blue",
                        level=logging.INFO,
                    )

                    # Send our info to node
                    message = json.dumps(
                        (None, self.role, self.public_key, self.rsa_pub_key.decode())
                    )
                    sock.send(message.encode())

                    # Perform handshake and ensure socket is closed after use
                    success = self._handshake(sock, (host, port), instigator=True)
                    sock.close()  # Close the socket after the handshake
                    return success

                except Exception as e:
                    wait_time = backoff * (2**attempt)  # Exponential backoff
                    self.debug_print(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. Retrying in {wait_time}s",
                        level=logging.WARNING,
                    )
                    time.sleep(wait_time)
                    self.remove_port_mapping(our_port)

        else:
            return False

    def bootstrap(self):
        """Bootstrap node to existing validators"""
        if self.off_chain_test is True:
            return

        candidates = []

        # Connect with some seed nodes from config file
        with open(os.path.join(CONFIG_PATH, "config.json"), "r") as file:
            _config = json.load(file)
            seed_validators = _config["network"]["mainnet"]["seeds"]

            for seed_validator in seed_validators:
                id_hash, host, port = seed_validator
                connected = self.connect_node(id_hash, host, port)
                if connected:
                    candidates.append(id_hash)
                else:
                    self._delete_item(id_hash)

        # Connect to additional randomly selected validators from the network
        n_validators = self.get_validator_count()
        sample_size = min(n_validators, 0)
        for i in [random.randint(1, n_validators) for _ in range(sample_size)]:
            # Random validator id
            validator_id = random.randrange(1, n_validators + 1)

            # Get key validator information from smart contract
            validator_contract_info = self.get_validator_info(validator_id)

            if validator_contract_info is not None:
                is_active, id_hash = validator_contract_info
                id_hash = id_hash.hex()
                validator_p2p_info = self.query_dht(id_hash)

                if validator_p2p_info is None:
                    self._delete_item(id_hash)
                    continue

                # Connect to the validator's node and exchange information
                # TODO what if we receive false connection info from validator: how to report?
                connected = self.connect_node(
                    id_hash, validator_p2p_info["host"], validator_p2p_info["port"]
                )

                if not connected:
                    self._delete_item(id_hash)
                    continue

                candidates.append(validator_id)

        return candidates

    def _init_sock(self) -> None:
        """Initializes the main socket for handling incoming connections."""
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Check for prior port usage (best to maintain same port for simple bootstrapping)
        port = get_key(".tensorlink.env", self.rsa_key_hash)
        if port is None or not port:
            # If no port is found, get the next available one
            port = self._get_next_port()

        port = int(port)
        self.port = int(port)

        if self.upnp:
            while True:
                result = self.add_port_mapping(
                    port, port
                )  # Forward the port using UPnP
                if result is False:
                    self.port += 1
                    port += 1
                elif result is True:
                    break
                else:
                    raise "Error binding port."

        self.sock.bind((self.host, port))
        self.sock.settimeout(3)
        self.sock.listen(5)

        # Update the port in the config if necessary
        set_key(".tensorlink.env", self.rsa_key_hash, str(port))

    def _init_upnp(self) -> None:
        """Enables UPnP on main socket to allow connections"""
        self.upnp = UPnP()
        self.upnp.discoverdelay = 2_000
        devices_found = self.upnp.discover()
        self.upnp.selectigd()

        # Clean up mappings previously created by this application.
        try:
            if devices_found == 0:
                self.debug_print("No UPnP devices found.")
                return

            self.clean_port_mappings()

        except Exception as e:
            self.debug_print(f"Error during UPnP cleanup: {e}")

        self.add_port_mapping(self.port, self.port)

    def _get_port_identifier(self):
        """Generate a unique identifier for this node's UPnP mappings"""
        # Use first 8 chars of hash to generate a unique identifier
        short_hash = self.rsa_key_hash[:8]
        return f"SmartNode-{short_hash}-{self.role}"

    def add_port_mapping(self, external_port, internal_port):
        """Open up a port via UPnP for a connection"""
        if self.upnp:
            try:
                port_identifier = self._get_port_identifier()
                result = self.upnp.addportmapping(
                    external_port,
                    "TCP",
                    self.upnp.lanaddr,
                    internal_port,
                    port_identifier,
                    "",
                )

                if result:
                    self.debug_print(
                        f"SmartNode -> UPnP port forward successful on port {self.port}"
                    )
                    return True
                else:
                    self.debug_print(
                        f"SmartNode -> Failed to initialize UPnP. (internal port: {internal_port},"
                        f" external port: {external_port})",
                        level=logging.CRITICAL,
                        colour="bright_red",
                    )
                    return False

            except Exception as e:
                if "ConflictInMapping" in str(e):
                    self.debug_print(
                        f"SmartNode -> Port {external_port} is already mapped.",
                        level=logging.DEBUG,
                    )
                    return False
                else:
                    raise e

    def remove_port_mapping(self, external_port):
        """Close a port via UPnP"""
        if self.upnp:
            try:
                # Attempt to delete the port mapping
                result = self.upnp.deleteportmapping(external_port, "TCP")

                if result is True:
                    self.debug_print(
                        f"SmartNode -> Successfully removed UPnP port mapping for external port {external_port}"
                    )
                else:
                    self.debug_print(
                        f"SmartNode -> Could not remove port mapping: {result}",
                        level=logging.WARNING,
                        colour="yellow",
                    )
            except Exception as e:
                self.debug_print(
                    f"SmartNode -> Error removing UPnP port mapping for port {external_port}: {e}",
                    level=logging.ERROR,
                    colour="bright_red",
                )

    def clean_port_mappings(self):
        """
        Lists all port mappings on the UPnP-enabled router.
        """
        mappings = []
        index = 38751

        if not self.upnp:
            self.debug_print(
                "SmartNode -> UPnP is not initialized.", level=logging.WARNING
            )
            return mappings

        while True:
            try:
                # Retrieve port mapping at the current index
                mapping = self.upnp.getspecificportmapping(index, "TCP")
                if mapping:
                    _, port, description, _, _ = mapping
                    if description != self._get_port_identifier():
                        self.remove_port_mapping(port)
                index += 1

            except Exception as e:
                # Stop when there are no more entries
                if "SpecifiedArrayIndexInvalid" in str(e):
                    break

                self.debug_print(
                    f"SmartNode -> Error retrieving port mapping at index {index}: {e}",
                    level=logging.ERROR,
                )
                break

            if index > 39_000:
                break

        return mappings

    def get_external_ip(self):
        """Get public IP address"""
        return self.upnp.externalipaddress()

    def _create_connection(
        self,
        connection: socket.socket,
        host: str,
        port: int,
        main_port: int,
        node_id: bytes,
        role: int,
    ) -> Connection:
        """Creates a connection thread object from connection.py for individual connections"""
        return Connection(self, connection, host, port, main_port, node_id, role)

    def _can_connect(self, host: str, port: int):
        """Makes sure we are not trying to connect to ourselves or a connected nodes"""
        # Check if trying to connect to self
        if host == self.host and port == self.port:
            self.debug_print(
                "SmartNode -> connect_with_node: cannot connect with yourself!"
            )
            return False

        # Check if already connected
        for node in self.nodes.values():
            if node.host == host and (node.port == port or node.main_port == port):
                self.debug_print(
                    f"SmartNode -> connect_with_node: already connected with node: {node.node_id}"
                )
                return False

        return True

    def send_to_node(self, n: Connection, data: bytes) -> None:
        """Send data to a connected nodes"""
        if n in self.nodes.values():
            self.debug_print(
                f"SmartNode -> send_to_node: Sending {len(data)} to node: {n.host}:{n.port}"
            )
            n.send(data)
        else:
            self.debug_print("SmartNode -> send_to_node: node not found!", colour="red")

    def send_to_node_from_file(self, n: Connection, file, tag):
        if n in self.nodes.values():
            n.send_from_file(file, tag)
        else:
            self.debug_print("SmartNode -> send_to_node: node not found!", colour="red")

    def handle_message(self, node: Connection, data) -> None:
        """Callback method to handles incoming data from connections"""
        self.debug_print(
            f"SmartNode -> handle_message from {node.host}:{node.port} -> {data.__sizeof__()/1e6}MB"
        )
        self.handle_data(data, node)

    def ping_node(self, n: Connection):
        """Measure latency nodes latency"""
        n.pinged = time.time()
        self.send_to_node(n, b"PING")

    def run(self):
        self.connection_listener = threading.Thread(target=self._listen, daemon=True)
        self.connection_listener.start()

    def disconnect_node(self, node_id: str):
        if node_id in self.nodes:
            node = self.nodes[node_id]

            if node_id in self.validators:
                self.validators.remove(node_id)
            elif node_id in self.users:
                self.users.remove(node_id)
            elif node_id in self.workers:
                self.workers.remove(node_id)

            node.terminate_flag.set()
            self.remove_port_mapping(node.port)
            del self.nodes[node_id]

    def close_connection(self, n: socket.socket, additional_info: str = None) -> None:
        message = "closing connection"
        if additional_info:
            message += f": {additional_info}"

        self.debug_print("SmartNode -> " + message, colour="red", level=logging.DEBUG)
        self.remove_port_mapping(n.getsockname()[1])
        n.close()

    def _stop_upnp(self) -> None:
        """Shuts down UPnP on port"""
        if self.upnp:
            self.clean_port_mappings()
            self.debug_print("SmartNode -> _stop_upnp: UPnP cleaned.")

    def stop(self) -> None:
        """Shut down nodes and all associated connections/threads"""
        self.debug_print("Node stopping...", colour="bright_yellow", level=logging.INFO)
        self.terminate_flag.set()

        try:
            self.sock.close()
        except Exception as e:
            self.debug_print(
                f"Error closing socket: {e}", colour="bright_red", level=logging.ERROR
            )

        for node in self.nodes.values():
            node.stop()

        for node in self.nodes.values():
            node.join()

        self._stop_upnp()
        clean()

        self.debug_print("Node stopped.", colour="bright_yellow", level=logging.INFO)

    """Methods to Interact with Flask Endpoints"""

    def get_self_info(self):
        data = {
            "id": self.rsa_key_hash,
            "validators": [k.decode() for k in self.validators],
            "workers": [k.decode() for k in self.workers],
            "users": [k.decode() for k in self.users],
        }
        return data

    def _get_next_port(self):
        port = self.port

        while True:
            if port not in self.used_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind((self.host, port))
                    sock.close()
                    self.used_ports.add(port)
                    return port

                except OSError:
                    port += random.randint(1, 50)

                except Exception as e:
                    raise e
            else:
                port += random.randint(1, 50)

    """Methods for Smart Contract Interactions"""

    def get_validator_count(self):
        """Get number of listed validators on Smart Nodes"""
        num_validators = self.contract.functions.validatorCounter().call()
        return num_validators - 1

    def get_validator_info(self, validator_ind: int):
        """Get validator info from Smart Nodes"""
        try:
            (is_active, pub_key_hash) = self.contract.functions.getValidatorInfo(
                validator_ind
            ).call()
            return is_active, pub_key_hash

        except Exception as e:
            self.debug_print(
                f"Validator with the ID {validator_ind} not found!.\nException: {e}"
            )
