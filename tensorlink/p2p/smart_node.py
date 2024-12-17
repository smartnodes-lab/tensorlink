from tensorlink.crypto.rsa import *
from tensorlink.p2p.connection import Connection

from logging.handlers import TimedRotatingFileHandler
from collections import defaultdict
from miniupnpc import UPnP
from web3 import Web3
from dotenv import get_key, set_key
import ipaddress
import threading
import requests
import logging
import hashlib
import random
import socket
import json
import time
import os


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
    "bright_white": "\033[97m"
}


# Grab smart contract information
base_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(base_dir, '../config')
SM_CONFIG_PATH = os.path.join(CONFIG_PATH, 'SmartnodesCore.json')
MS_CONFIG_PATH = os.path.join(CONFIG_PATH, 'SmartnodesMultiSig.json')

with open(os.path.join(CONFIG_PATH, "config.json"), "r") as f:
    config = json.load(f)
    CHAIN_URL = config["api"]["chain-url"]
    CONTRACT = config["api"]["core"]
    MULTI_SIG_CONTRACT = config["api"]["multi-sig"]

with open(SM_CONFIG_PATH, "r") as f:
    METADATA = json.load(f)
ABI = METADATA["abi"]

with open(MS_CONFIG_PATH, "r") as f:
    MS_METADATA = json.load(f)
MULTI_SIG_ABI = MS_METADATA["abi"]

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


SNO_EVENT_SIGNATURES = {
    "JobRequest": "JobRequested(uint256,uint256,address[])",
    "JobComplete": "JobCompleted(uint256,uint256)",
    "JobDispute": "JobDisputed(uint256,uint256)",
    "ProposalCreated": "ProposalCreated(uint256,bytes)",
    "ProposalExecuted": "ProposalExecuted(uint256)"
}


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
        response = requests.get('https://api.ipify.org')
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
        debug_colour=None
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
        self.upnp = None
        self.nodes = {}  # node hash: Connection
        self.rate_limit = defaultdict(lambda: {"attempts": 0, "last_attempt": 0, "blocked_until": 0})
        self.max_attempts_per_minute = 5
        self.block_duration = 600

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
        self.id = 0

        # Stores key of stored values
        self.validators = []
        self.workers = []
        self.users = []
        self.jobs = []

        self.sno_events = {name: Web3.keccak(text=sig).hex() for name, sig in SNO_EVENT_SIGNATURES.items()}
        self.off_chain_test = off_chain_test
        self.local_test = local_test

        if upnp:
            self.init_upnp()

        self.init_sock()

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
                self.debug_print(f"SmartNode -> Could not retrieve contract: {e}", colour="bright_red",
                                 level=logging.CRITICAL)
                self.stop()

    def handle_data(self, data: bytes, node: Connection) -> bool:
        """
        Handles incoming data from nodes connections and performs an appropriate response.
        Each chunk of data has a tag at the beginning which defines the message type.
        """
        try:
            # Used to log ghost messages: messages that require some pre-context to have
            # received, but we don't have the context
            ghost = 0

            # Larger data streams are stored in secondary storage during streaming to improve efficiency
            if b"DONE STREAM" == data[:11]:
                file_name = (
                    f"tmp/streamed_data_{node.host}_{node.port}_{self.host}_{self.port}"
                )

                # Instead of loading file we can keep if need. This is gross code and should be changed
                if b"MODULE" in data[11:]:
                    streamed_bytes = data[11:]
                    os.rename(file_name, data[17:].decode())

                elif b"PARAMETERS" in data[11:]:
                    streamed_bytes = data[11:]
                    os.rename(file_name, "tmp/" + data[21:].decode() + "_parameters")

                else:
                    with open(file_name, "rb") as file:
                        streamed_bytes = file.read()
                        os.remove(file_name)

                self.handle_data(streamed_bytes, node)

            # We received a ping, send a pong
            elif b"PING" == data[:4]:
                self.send_to_node(node, b"PONG")

            # We received a pong, update latency
            elif b"PONG" == data[:4]:
                if node.pinged > 0:
                    node.stats["ping"] = time.time() - node.pinged
                    node.pinged = -1
                else:
                    self.debug_print(f"SmartNode -> Received pong with no ping (suspicious?)", colour="red")
                    ghost += 1

            elif b"REQUEST-VALUE-RESPONSE" == data[:22]:
                # Retrieve the value associated with the key from the DHT
                self.debug_print(f"SmartNode -> node ({node.host}:{node.port}) returned value.")

                # Not enough data received for specific request
                if len(data) < 86:
                    self.debug_print("SmartNode -> Received random chunk of data (small packet!)",
                                     colour="red")
                    # TODO not enough data received!
                    pass

                elif node.node_id in self.requests:
                    value_id = data[22:86].decode()

                    # We have received data that we have requested
                    if "REQUEST-VALUE" + value_id in self.requests[node.node_id]:
                        value = json.loads(data[86:])
                        self.requests[node.node_id].remove("REQUEST-VALUE" + value_id)
                        self._store_request(value_id, value)
                    else:
                        self.debug_print(f"SmartNode -> Received ghost data from node: {node.node_id}",
                                         colour="red")
                        # Report being sent data we have not requested
                        ghost += 1
                else:
                    # Report being sent data we have not requested
                    self.debug_print("SmartNode -> Received ghost data from unknown node!", colour="red")
                    ghost += 1

            # We have received a request for some data
            elif b"REQUEST-VALUE" == data[:13]:
                # TODO Check of how many requests they have sent recently to prevent spam
                self.debug_print(f"SmartNode -> node ({node.host}:{node.port}) requested value.", colour="blue")
                value = None
                value_hash = None

                if len(data) < 141:
                    # TODO not enough data received!
                    pass
                else:
                    value_hash = data[13:77].decode()
                    requester = data[77:141].decode()

                    # Get nodes info
                    value = self.query_dht(value_hash, requester)

                # Send back response
                self.send_to_node(
                    node,
                    b"REQUEST-VALUE-RESPONSE"
                    + value_hash.encode()
                    + json.dumps(value).encode()
                )

            # No recognized tag
            else:
                # We do not log a ghost here since SmartNode is meant to be a super class and this should
                # only be invoked by a super call
                return False

            # If ghost was detected
            if ghost > 0:
                node.ghosts += ghost
                # TODO: potentially some form of reporting mechanism via ip and port

            return True

        except Exception as e:
            if "[Errno 2] No such file or directory:" in e:
                print(1)

            self.debug_print(f"SmartNode -> Error handling data: {e}", colour="bright_red",
                             level=logging.ERROR)

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
    def query_dht(
        self, key_hash, requester: str = None, ids_to_exclude: list = None
    ):
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
                self.__delete(closest_node[0])

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
        key_hash: str,
        node: Connection,
        requester: bytes = None,
        ids_to_exclude: list = None,
    ):
        """Query a specific nodes for a value"""
        if requester is None:
            requester = self.rsa_key_hash

        if node.terminate_flag.is_set():
            return

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
            n_validators = self.get_validator_count()

            while replicate > 0:
                random_id = random.randrange(1, n_validators + 1)

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

    def __delete(self, key: str):
        """
        Delete a key-value pair from the DHT.
        """

        if key in self.routing_table:
            val = self.routing_table[key]
            bucket_index = self.calculate_bucket_index(key)
            bucket = self.buckets[bucket_index]

            if key in self.nodes or self.jobs:
                # Do not delete information related to active connections or jobs
                return

            del self.routing_table[key]
            self.debug_print(f"SmartNode -> Key {key} deleted from DHT.", colour="blue")
        else:
            self.debug_print(f"SmartNode -> Key {key} not found in DHT.", colour="red", level=logging.ERROR)

    def calculate_bucket_index(self, key: str):
        """
        Find the index of a bucket given the key
        """
        key_int = int(key.encode(), 16)
        bucket_index = key_int % len(self.buckets)

        return bucket_index

    def is_blocked(self, ip_address):
        """Check if an IP address is currently blocked."""
        current_time = int(time.time())
        block_info = self.rate_limit[ip_address]
        if block_info["blocked_until"] > current_time:
            return True
        return False

    def record_attempt(self, ip_address):
        """Record a connection attempt for an IP address."""
        current_time = time.time()
        block_info = self.rate_limit[ip_address]

        # Reset the attempt count if the last attempt was over a minute ago
        if current_time - block_info["last_attempt"] > 60:
            self.rate_limit[ip_address] = {"attempts": 0, "last_attempt": 0, "blocked_until": 0}

        self.rate_limit[ip_address]["attempts"] += 1
        self.rate_limit[ip_address]["last_attempt"] = current_time

        # Block the IP if it exceeds the limit
        if self.rate_limit[ip_address]["attempts"] > self.max_attempts_per_minute:
            self.rate_limit[ip_address]["blocked_until"] = current_time + self.block_duration

    """Peer-to-peer methods"""
    def listen(self):

        """Listen for incoming connections and initialize custom handshake"""
        while not self.terminate_flag.is_set():
            try:
                if self.sock.fileno() == -1:
                    return

                # Unpack nodes info
                connection, node_address = self.sock.accept()
                ip_address = node_address[0]

                # Check rate limiting
                if self.is_blocked(ip_address):
                    self.close_connection(
                        connection,
                        f"listen: connection refused, rate limit exceeded for {ip_address}",
                    )
                    continue

                self.record_attempt(ip_address)

                # Attempt custom handshake
                if self.max_connections == 0 or len(self.nodes) < self.max_connections:
                    self.handshake(connection, node_address)

                else:
                    self.close_connection(
                        connection,
                        "listen: connection refused, max connections reached",
                    )

            except socket.timeout:
                # self.debug_print(f"listen: connection timed out!")
                pass

            except Exception as e:
                self.debug_print(f"SmartNode -> listen connection error {e}", colour="bright_red",
                                 level=logging.CRITICAL)

            # self.reconnect_nodes()

    def handshake(
        self, connection: socket.socket, node_address, instigator=False
    ) -> bool:
        """Validates incoming node's keys via a random number swap, along with SC verification of connecting user"""
        id_info = connection.recv(1024)
        _, role, id_no, connected_node_id = json.loads(id_info)
        connected_node_id = connected_node_id.encode()
        node_id_hash = hashlib.sha256(connected_node_id).hexdigest()

        # Query node info/history from dht for reputation if we're already connected to network
        if len(self.nodes) > 0:
            # Get node information from any other node
            node_info = self.query_dht(node_id_hash, ids_to_exclude=[node_id_hash])

            if node_info:
                if node_info["reputation"] == 0:
                    self.debug_print(
                        f"SmartNode -> User with poor reputation attempting connection: {node_id_hash}",
                        colour="red", level=logging.WARNING
                    )
                    connection.close()

        # Check that we have been assigned to the user by a validator
        if role == "U" and self.role == "W":
            if node_id_hash not in self.requests:
                self.close_connection(connection, f"Was not assigned to user: {node_id_hash}")
                return False

        # If we are the instigator of the connection, we will have received a request to verify our id
        if instigator:
            encrypted_number = _.encode()
            proof = decrypt(encrypted_number, self.role)

            try:
                proof = float(proof)

            except Exception as e:
                self.close_connection(
                    connection, f"Proof request was not valid: {e}"
                )
                return False

        # Confirm their key is a valid RSA key
        if authenticate_public_key(connected_node_id):
            # Role-specific confirmations (Must be U, W, or V to utilize smart nodes, tensorlink, etc.)
            if self.off_chain_test is False:
                try:

                    if role == "V":
                        # Query contract for users key hash
                        is_active, pub_key_hash, wallet_address = self.contract.functions.getValidatorInfo(
                            id_no
                        ).call()

                        # If validator was not found
                        if not is_active or node_id_hash != bytes.hex(pub_key_hash):
                            # TODO: potentially some form of reporting mechanism via ip and port
                            self.close_connection(
                                connection,
                                f"Validator {connected_node_id} not listed on SmartnodesCore!"
                            )

                    elif role == "W" or role == "U":
                        # TODO: worker/user handling
                        pass

                    else:
                        # TODO: potentially some form of reporting mechanism via ip and port
                        self.close_connection(connection,
                                              f"SmartNode -> connection refused, invalid role: {node_address}")
                        raise f"listen: connection refused, invalid role: {node_address}"

                except Exception as e:
                    self.close_connection(
                        connection, f"handshake: contract query error: {e}"
                    )

            # Random number swap to confirm the nodes RSA key
            rand_n = random.random()
            encrypted_number = encrypt(str(rand_n).encode(), self.role, connected_node_id)

            # Encrypt random number with nodes's key to confirm their identity
            # If we are the instigator, we will also need to send our proof
            if instigator:
                # Send main port if we are the instigator
                port = connection.getsockname()[1]
                message = json.dumps((self.port, proof, encrypted_number.decode()))
            else:
                message = json.dumps(
                    (encrypted_number.decode(), self.role, self.id, self.rsa_pub_key.decode())
                )

            connection.send(message.encode())

            # Await response
            response = connection.recv(1024)

            if instigator:
                # We have already verified ours and just want to confirm theirs
                try:
                    new_port, rand_n_proof = json.loads(response)
                    rand_n_proof = float(rand_n_proof)
                    main_port = node_address[1]

                    # Select a new port for the node to use if we are not the instigator
                    our_new_port = self.get_next_port()
                    self.add_port_mapping(our_new_port, our_new_port)
                    connection.close()

                    self.debug_print(f"SmartNode -> Selected next port: {our_new_port} for new connection")
                    self.debug_print(f"SmartNode -> Switching connection to the new port: {node_address[0]}:{new_port}")

                    # Establish a new connection to the node on the provided port
                    new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    new_sock.bind((self.host, our_new_port))
                    new_sock.connect((node_address[0], new_port))
                    new_sock.settimeout(3)
                    connection = new_sock

                except socket.error as e:
                    self.close_connection(connection, f"Socket error during connection switch: {e}")
                    return False

                except Exception as e:
                    self.close_connection(
                        connection, f"Proof was not valid: {e}"
                    )
                    return False
            else:
                # Unpack response (verification of their ID along with a request to verify ours)
                response = json.loads(response)
                main_port, rand_n_proof, verification = response
                verification = decrypt(verification.encode(), self.role)

                # Select a new port for the node to use if we are not the instigator
                our_new_port = self.get_next_port()
                self.debug_print(f"SmartNode -> Selected next port: {our_new_port} for new connection")
                self.add_port_mapping(our_new_port, our_new_port)

                # Send the new port and proof of random number
                response = json.dumps((our_new_port, verification.decode()))
                connection.send(response.encode())

                # Close the current connection and listen on the new port
                connection.close()
                self.debug_print(f"SmartNode -> Listening for the instigator on the new port: {our_new_port}")

                # Create a new socket and bind to the selected port
                new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                new_sock.bind((self.host, our_new_port))
                new_sock.settimeout(3)
                new_sock.listen(3)

                # Accept the incoming connection on the new port
                connection, new_node_address = new_sock.accept()

            # If the nodes has confirmed his identity
            if rand_n_proof == rand_n:
                # Check to see if we are already connected
                for node in self.nodes.values():
                    if node.host == node_address[0] and node.port == our_new_port:
                        self.debug_print(
                            f"SmartNode -> connect_with_node: already connected with nodes: {node.node_id}"
                        )
                        connection.close()
                        return False

                # Check network for existing info on node
                node_info = self.query_dht(node_id_hash)

                if node_info:
                    if node_info["reputation"] < 50:
                        self.debug_print(
                            f"SmartNode -> connect_with_node: poor reputation: {node.node_id}",
                            colour="red",
                            level=logging.WARNING
                        )
                        connection.close()
                        return False

                thread_client = self.create_connection(
                    connection,
                    node_address[0],
                    new_port if instigator else node_address[1],
                    main_port,
                    connected_node_id,
                    role,
                )
                thread_client.start()

                # Finally connect to the node
                if not thread_client.terminate_flag.is_set():
                    if node_info is None:
                        node_info = get_connection_info(
                            thread_client,
                            main_port=main_port if not instigator else None,
                            upnp=False if self.upnp is None else True
                        )

                    self.nodes[node_id_hash] = thread_client
                    self.store_value(node_id_hash, node_info)

                    if role == "V":
                        self.validators.append(node_id_hash)

                    elif role == "W":
                        self.workers.append(node_id_hash)

                    elif role == "U":
                        self.users.append(node_id_hash)

                        # If we are connecting to a user (for a job), boost connection speed
                        if self.role == "W":
                            thread_client.adjust_chunk_size("large")    

                    self.debug_print(
                        f"SmartNode -> Connected to node: {thread_client.host}:{thread_client.port}",
                        level=logging.INFO,
                        colour="bright_green"
                    )
                    return True

                else:
                    return False

            else:
                self.close_connection(connection, "Proof request was not valid.")
                return False

        else:
            self.close_connection(connection, "RSA key was not valid.")
            return False

    def connect_node(
            self, id_hash: bytes, host: str, port: int, reconnect: bool = False
    ) -> bool:
        """
        Connect to a role and exchange information to confirm its role in the Smart Nodes network.
        """
        can_connect = self.can_connect(host, port)

        # Check that we are not already connected
        if id_hash in self.nodes:
            self.debug_print(f"SmartNode -> connect_node: Already connected to {id_hash}")
            return True

        if can_connect:
            for attempt in range(2):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    our_port = self.get_next_port()
                    self.add_port_mapping(our_port, our_port)
                    self.debug_print(f"SmartNode -> Selected next port: {our_port} for new connection")

                    sock.bind((self.host, our_port))
                    sock.connect((host, port))
                    sock.settimeout(10)

                    self.debug_print(f"SmartNode -> connect_node: connecting to {host}:{port}", colour="blue",
                                     level=logging.INFO)
                    message = json.dumps((None, self.role, self.id, self.rsa_pub_key.decode()))
                    sock.send(message.encode())

                    # Perform handshake and ensure socket is closed after use
                    success = self.handshake(sock, (host, port), instigator=True)
                    sock.close()  # Close the socket after the handshake
                    return success

                except Exception as e:
                    self.debug_print(f"Attempt {attempt + 1}: could not connect to {host}:{port} -> {e}",
                                     colour="red", level=logging.WARNING)
                    time.sleep(1)
                    self.remove_port_mapping(our_port)
                    if attempt == 2:  # Last attempt
                        return False

        else:
            return False

    def bootstrap(self):
        """Bootstrap node to existing validators"""
        if self.off_chain_test is True:
            return

        n_validators = self.get_validator_count()
        sample_size = min(n_validators, 1)
        candidates = []

        with open(os.path.join(CONFIG_PATH, "config.json"), "r") as file:
            _config = json.load(file)
            seed_validators = _config["network"]["mainnet"]["seeds"]
            for seed_validator in seed_validators:
                id_hash, host, port = seed_validator
                connected = self.connect_node(id_hash, host, port)
                if connected:
                    candidates.append(id_hash)
                else:
                    self.__delete(id_hash)

        # # Connect to randomly selected  validators
        # for i in [random.randint(1, n_validators) for _ in range(sample_size)]:
        #     # Random validator id
        #     validator_id = random.randrange(1, n_validators + 1)
        #
        #     # Get key validator information from smart contract
        #     validator_contract_info = self.get_validator_info(validator_id)
        #
        #     if validator_contract_info is not None:
        #         is_active, id_hash = validator_contract_info
        #         id_hash = id_hash.hex()
        #         validator_p2p_info = self.query_dht(id_hash)
        #
        #         if validator_p2p_info is None:
        #             self.__delete(id_hash)
        #             continue
        #
        #         # Connect to the validator's node and exchange information
        #         # TODO what if we receive false connection info from validator: how to report?
        #         connected = self.connect_node(
        #             id_hash, validator_p2p_info["host"], validator_p2p_info["port"]
        #         )
        #
        #         if not connected:
        #             self.__delete(id_hash)
        #             continue
        #
        #         candidates.append(validator_id)

        return candidates

    def init_sock(self) -> None:
        """Initializes the main socket for handling incoming connections."""
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Check for prior port usage (best to maintain same port for simple bootstrapping)
        port = get_key(".env", self.rsa_key_hash)
        if port is None or not port:
            # If no port is found, get the next available one
            port = self.get_next_port()

        port = int(port)
        self.port = int(port)

        if self.upnp:
            while True:
                result = self.add_port_mapping(port, port)  # Forward the port using UPnP
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
        set_key(".env", self.rsa_key_hash, str(port))

    def init_upnp(self) -> None:
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

    def add_port_mapping(self, external_port, internal_port):
        if self.upnp:
            try:
                result = self.upnp.addportmapping(
                    external_port, "TCP", self.upnp.lanaddr, internal_port, "SmartNode", ""
                )

                if result:
                    self.debug_print(f"SmartNode -> UPnP port forward successful on port {self.port}")
                    return True
                else:
                    self.debug_print(f"SmartNode -> Failed to initialize UPnP. (internal port: {internal_port},"
                                     f" external port: {external_port})", level=logging.CRITICAL, colour="bright_red")
                    return False

            except Exception as e:
                if "ConflictInMapping" in str(e):
                    self.debug_print(f"SmartNode -> Port {external_port} is already mapped.", level=logging.DEBUG)
                    return False
                else:
                    raise e

    def remove_port_mapping(self, external_port):
        if self.upnp:
            try:
                # Attempt to delete the port mapping
                result = self.upnp.deleteportmapping(external_port, "TCP")

                if result is True:  # Some UPnP implementations return None on success
                    self.debug_print(
                        f"SmartNode -> Successfully removed UPnP port mapping for external port {external_port}")
                else:
                    self.debug_print(f"SmartNode -> Could not remove port mapping: {result}",
                                     level=logging.WARNING, colour="yellow")
            except Exception as e:
                self.debug_print(f"SmartNode -> Error removing UPnP port mapping for port {external_port}: {e}",
                                 level=logging.ERROR, colour="bright_red")

    def clean_port_mappings(self):
        """
        Lists all port mappings on the UPnP-enabled router.
        """
        mappings = []
        index = 38751

        if not self.upnp:
            self.debug_print("SmartNode -> UPnP is not initialized.", level=logging.WARNING)
            return mappings

        while True:
            try:
                # Retrieve port mapping at the current index
                mapping = self.upnp.getspecificportmapping(index, "TCP")
                if mapping:
                    _, port, description, _, _ = mapping
                    if description == "SmartNode":
                        self.remove_port_mapping(port)
                index += 1

            except Exception as e:
                # Stop when there are no more entries
                if "SpecifiedArrayIndexInvalid" in str(e):
                    break
                self.debug_print(f"SmartNode -> Error retrieving port mapping at index {index}: {e}",
                                 level=logging.ERROR)
                break

            if index > 39_000:
                break

        return mappings

    def get_external_ip(self):
        """Get public IP address"""
        return self.upnp.externalipaddress()

    def create_connection(
        self,
        connection: socket.socket,
        host: str,
        port: int,
        main_port: int,
        node_id: bytes,
        role: int,
    ) -> Connection:
        return Connection(self, connection, host, port, main_port, node_id, role)

    def can_connect(self, host: str, port: int):
        """Makes sure we are not trying to connect to ourselves or a connected nodes"""
        # Check if trying to connect to self
        if host == self.host and port == self.port:
            self.debug_print("SmartNode -> connect_with_node: cannot connect with yourself!")
            return False

        # Check if already connected
        for node in self.nodes.values():
            if node.host == host and node.port == port:
                self.debug_print(
                    f"SmartNode -> connect_with_node: already connected with node: {node.node_id}"
                )
                return False

        return True

    def send_to_node(
        self, n: Connection, data: bytes
    ) -> None:
        """Send data to a connected nodes"""
        if n in self.nodes.values():
            self.debug_print(f"SmartNode -> send_to_node: Sending {len(data)} to node: {n.host}:{n.port}")
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
        self.connection_listener = threading.Thread(target=self.listen, daemon=True)
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

    def close_connection(
        self, n: socket.socket, additional_info: str = None
    ) -> None:
        message = "closing connection"
        if additional_info:
            message += f": {additional_info}"

        self.debug_print("SmartNode -> " + message, colour="red", level=logging.DEBUG)
        self.remove_port_mapping(n.getsockname()[1])
        n.close()

    def stop_upnp(self) -> None:
        """Shuts down UPnP on port"""
        if self.upnp:
            self.clean_port_mappings()
            self.debug_print(f"SmartNode -> stop_upnp: UPnP cleaned.")

    def stop(self) -> None:
        """Shut down nodes and all associated connections/threads"""
        self.debug_print(f"Node stopping...", colour="bright_yellow", level=logging.INFO)
        self.terminate_flag.set()

        try:
            self.sock.close()
        except Exception as e:
            self.debug_print(f"Error closing socket: {e}", colour="bright_red", level=logging.ERROR)

        for node in self.nodes.values():
            node.stop()

        for node in self.nodes.values():
            node.join()

        self.stop_upnp()
        clean()

        self.debug_print("Node stopped.", colour="bright_yellow", level=logging.INFO)

    """Methods to Interact with Flask Endpoints"""
    def get_self_info(self):
        data = {
            "id": self.rsa_key_hash.decode(),
            "validators": [k.decode() for k in self.validators],
            "workers": [k.decode() for k in self.workers],
            "users": [k.decode() for k in self.users],
        }
        return data

    def get_next_port(self):
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
            is_active, pub_key_hash, wallet_address = self.contract.functions.getValidatorInfo(
                validator_ind
            ).call()
            return is_active, pub_key_hash

        except Exception as e:
            self.debug_print(f"Validator with the ID {validator_ind} not found!.\nException: {e}")
