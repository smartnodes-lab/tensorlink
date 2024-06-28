from src.cryptography.rsa import *
from src.p2p.connection import Connection

from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv
from typing import Callable
from miniupnpc import UPnP
from web3 import Web3
import threading
import logging
import hashlib
import pickle
import random
import socket
import json
import time
import os


load_dotenv()

# Grab smart contract information
CHAIN_URL = os.getenv("CHAIN_URL")
CONTRACT = os.getenv("CONTRACT")

with open("./config/SmartNodes.json", "r") as f:
    METADATA = json.load(f)

ABI = METADATA["abi"]
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# Configure logging with TimedRotatingFileHandler
log_handler = TimedRotatingFileHandler(
    "dht_logs.log", when="midnight", interval=1, backupCount=30
)
log_handler.setFormatter(logging.Formatter("%(message)s"))
log_handler.suffix = "%Y%m%d"
logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.INFO)
STATE_FILE = "./dht_state.json"
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
    Calculate the XOR distance between a key and a node ID.
    """
    return int(key_hash, 16) ^ int(node_id, 16)


def get_connection_info(node, main_port=None):
    """Connection info for routing table storage"""
    info = {
        "host": node.host,
        "port": node.port if not main_port else main_port,
        "role": node.role,
        "id": node.node_id,
        "reputation": node.reputation,
    }
    return info


def log_entry(node, metadata):
    entry = {"timestamp": time.ctime(), "node": node.node_id, "metadata": metadata}
    logging.info(json.dumps(entry))


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
    A P2P node secured by RSA encryption and smart contract validation for the Smartnodes ecosystem.
    """

    def __init__(
        self,
        url: str = CHAIN_URL,
        contract: str = CONTRACT,
        debug: bool = False,
        max_connections: int = 0,
        upnp: bool = True,
        off_chain_test: bool = False,
        debug_colour=None
    ):
        super(SmartNode, self).__init__()

        # Node Parameters
        self.terminate_flag = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Get private ip
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        self.host = s.getsockname()[0]
        s.close()

        self.port = BASE_PORT
        self.used_ports = set()
        self.debug = debug
        self.max_connections = max_connections

        self.debug_colour = None
        if debug_colour:
            self.debug_colour = debug_colour

        self.upnp = None
        self.off_chain_test = off_chain_test
        self.nodes = {}  # node-hash: Connection
        self.node_stats = (
            {}
        )  # node-hash: {message counts & other key stats to keep track of}

        # DHT Parameters
        self.replication_factor = 3
        self.bucket_size = 2
        self.buckets = [Bucket(d, self.bucket_size) for d in range(256)]
        self.routing_table = {}
        self.requests = {}

        # More parameters for smart contract / p2p info
        self.rsa_pub_key = None
        self.rsa_key_hash = None
        self.role = b""
        self.id = 0

        # Stores key of stored values
        self.workers = []
        self.validators = []
        self.users = []
        self.jobs = []

        if upnp:
            self.init_upnp()

        self.init_sock()

        if self.off_chain_test is False:
            # Smart node parameters for additional security and contract connectivity
            self.url = url
            self.chain = Web3(Web3.HTTPProvider(url))
            self.contract_address = Web3.to_checksum_address(contract)

            # Grab the SmartNode contract
            try:
                self.contract = self.chain.eth.contract(
                    address=self.contract_address, abi=ABI
                )

            except Exception as e:
                self.debug_print(f"Could not retrieve contract: {e}")
                self.stop()

    def handle_data(self, data: bytes, node: Connection) -> bool:
        """
        Handles incoming data from node connections and performs an appropriate response.
        Each chunk of data has a tag at the beginning which defines the message type.
        """
        try:
            # Used to log ghost messages: messages that require some pre-context to have
            # received, but we don't have the context
            ghost = 0

            # Larger data streams are stored in secondary storage during streaming to improve efficiency
            if b"DONE STREAM" == data[:11]:
                file_name = (
                    f"streamed_data_{node.host}_{node.port}_{self.host}_{self.port}"
                )

                with open(file_name, "rb") as file:
                    streamed_bytes = file.read()

                self.handle_data(streamed_bytes, node)
                os.remove(file_name)

            # We received a ping, send a pong
            elif b"PING" == data[:4]:
                self.update_node_stats(node.node_id, "PING")
                self.send_to_node(node, b"PONG")

            # We received a pong, update latency
            elif b"PONG" == data[:4]:
                if node.pinged > 0:
                    node.ping = time.time() - node.pinged
                    node.pinged = -1
                else:
                    self.debug_print(f"Received pong with no ping (suspicious?)")
                    ghost += 1

            elif b"REQUEST-VALUE-RESPONSE" == data[:22]:
                # Retrieve the value associated with the key from the DHT
                self.debug_print(
                    f"handle_data: node ({node.host}:{node.port}) returned value."
                )

                # Not enough data received for specific request
                if len(data) < 86:
                    # TODO not enough data received!
                    pass

                elif node.node_id in self.requests:
                    value_id = data[22:86]

                    # We have received data that we have requested
                    if value_id in self.requests[node.node_id]:
                        value = pickle.loads(data[86:])
                        self.requests[node.node_id].remove(value_id)
                        self.routing_table[value_id] = value
                    else:
                        # Report being sent data we have not requested
                        ghost += 1
                else:
                    # Report being sent data we have not requested
                    ghost += 1

            # We have received a request for some data
            elif b"REQUEST-VALUE" == data[:13]:
                # TODO Check of how many requests they have sent recently to prevent spam
                self.debug_print(
                    f"handle_data: node ({node.host}:{node.port}) requested value."
                )
                validator_info = None
                value_hash = None

                if len(data) < 141:
                    # TODO not enough data received!
                    pass
                else:
                    value_hash = data[13:77]
                    requester = data[77:141]

                    # Get node info
                    validator_info = self.query_dht(value_hash, requester)

                # Send back response
                self.send_to_node(
                    node,
                    b"REQUEST-VALUE-RESPONSE"
                    + value_hash
                    + pickle.dumps(None if validator_info is None else validator_info),
                )

            # No recognized tag
            else:
                # We do not log a ghost here since SmartNode is meant to be a super class and this should
                # only be invoked by a super call
                return False

            # If ghost was detected
            if ghost > 0:
                self.update_node_stats(node.node_id, "GHOST")
                # TODO: potentially some form of reporting mechanism via ip and port

            return True

        except Exception as e:
            self.debug_print(f"handle_data: Error handling data: {e}")

    def debug_print(self, message) -> None:
        """Print to console if debug is enabled"""
        if self.debug:
            if self.debug_colour is None:
                print(f"{self.host}:{self.port} -> {message}")
            else:
                print(f"{self.debug_colour}{self.host}:{self.port} -> {message}")

    def listen(self):
        """Listen for incoming connections and initialize custom handshake"""
        while not self.terminate_flag.is_set():
            try:
                # Unpack node info
                connection, node_address = self.sock.accept()

                # Attempt custom handshake
                if self.max_connections == 0 or len(self.nodes) < self.max_connections:
                    self.handshake(connection, node_address)

                else:
                    self.close_connection_socket(
                        connection,
                        "listen: connection refused, max connections reached",
                    )

            except socket.timeout:
                # self.debug_print(f"listen: connection timeout!")
                pass

            except Exception as e:
                self.debug_print(f"listen: connection error {e}")

            # self.reconnect_nodes()

    def handshake(
        self, connection: socket.socket, node_address, instigator=False
    ) -> bool:
        """Validates incoming node's keys via a random number swap, along with SC verification of connecting user"""
        id_info = connection.recv(4096)
        _, role, id_no, connected_node_id = pickle.loads(id_info)
        node_id_hash = hashlib.sha256(connected_node_id).hexdigest().encode()

        # Query node info/history from dht for reputation if we are a validator
        if self.role == b"V":
            if len(self.nodes) > 0:
                node_info = self.query_dht(node_id_hash)
                if node_info:
                    if node_info["reputation"] == 0:
                        self.debug_print(
                            f"User with poor reputation attempting connection: {node_id_hash}"
                        )
                        connection.close()

        # If we are the instigator of the connection, we will have received a request to verify our id
        if instigator:
            encrypted_number = _
            proof = decrypt(encrypted_number, self.role)

            try:
                proof = float(proof)

            except Exception as e:
                self.close_connection_socket(
                    connection, f"Proof request was not valid: {e}"
                )
                return False

        # Confirm their key is a valid RSA key
        if authenticate_public_key(connected_node_id):
            # Role-specific confirmations (Must be U, W, or V to utilize smart nodes, tensorlink, etc.)
            if self.off_chain_test is False:
                if role == b"V":
                    try:
                        # Query contract for users key hash
                        validator_info = self.contract.functions.getValidatorInfo(
                            id_no
                        ).call()

                        # If validator was not found
                        if not validator_info[0]:
                            self.close_connection_socket(
                                connection,
                                f"handshake: validator role claimed but is not "
                                f"listed on Smart Nodes!: {connected_node_id}",
                            )
                            self.update_node_stats(node_id_hash, "GHOST")
                            # TODO: potentially some form of reporting mechanism via ip and port
                            return False

                    except Exception as e:
                        self.close_connection_socket(
                            connection, f"handshake: contract query error: {e}"
                        )

                elif role == b"U":
                    # TODO: user handling, to be done once users are required to register (post-alpha)
                    pass

                elif role == b"W":
                    # TODO: worker handling
                    pass

                else:
                    # TODO: potentially some form of reporting mechanism via ip and port
                    self.close_connection_socket(
                        connection,
                        f"listen: connection refused, invalid role: {node_address}",
                    )

            # Random number swap to confirm the nodes RSA key
            rand_n = random.random()
            encrypted_number = encrypt(str(rand_n).encode(), self.role, connected_node_id)

            # Encrypt random number with node's key to confirm their identity
            # If we are the instigator, we will also need to send our proof
            if instigator:
                # Send main port if we are the instigator
                message = pickle.dumps((self.port, proof, encrypted_number))
            else:
                message = pickle.dumps(
                    (encrypted_number, self.role, self.id, self.rsa_pub_key)
                )

            connection.send(message)

            # Await response
            response = connection.recv(4096)

            if instigator:
                # We have already verified ours and just want to confirm theirs
                try:
                    rand_n_proof = float(response)
                    main_port = node_address[1]

                except Exception as e:
                    self.close_connection_socket(
                        connection, f"Proof was not valid: {e}"
                    )
                    return False
            else:
                # Unpack response (verification of their ID along with a request to verify ours)
                response = pickle.loads(response)
                main_port, rand_n_proof, verification = response
                verification = decrypt(verification, self.role)

                # Send our verification (their random number request)
                connection.send(verification)

            # If the node has confirmed his identity
            if rand_n_proof == rand_n:
                # If ID is confirmed, we solidify the connection
                thread_client = self.create_connection(
                    connection,
                    node_address[0],
                    node_address[1],
                    main_port,
                    connected_node_id,
                    role,
                )
                thread_client.start()

                # Check to see if we are already connected
                for node in self.nodes.values():
                    if node.host == node_address[0] and node.port == node_address[1]:
                        self.debug_print(
                            f"connect_with_node: already connected with node: {node.node_id}"
                        )
                        self.close_connection(thread_client)
                        return False

                # Finally connect to the node
                if not thread_client.terminate_flag.is_set():
                    self.debug_print(
                        f"Connected to node: {thread_client.host}:{thread_client.port}"
                    )
                    self.nodes[node_id_hash] = thread_client

                    self.store_value(
                        node_id_hash,
                        get_connection_info(
                            thread_client, main_port if not instigator else None
                        ),
                    )

                    if role == b"V":
                        self.validators.append(node_id_hash)
                    elif role == b"W":
                        self.workers.append(node_id_hash)
                    elif role == b"U":
                        self.users.append(node_id_hash)

                    return True

                else:
                    return False

            else:
                self.close_connection_socket(connection, "Proof request was not valid.")
                return False

        else:
            self.close_connection_socket(connection, "RSA key was not valid.")
            return False

    def connect_node(
        self, id_hash: bytes, host: str, port: int, reconnect: bool = False
    ) -> bool:
        """
        Connect to a node and exchange information to confirm its role in the Smart Nodes network.
        """
        can_connect = self.can_connect(host, port)

        # Check that we are not already connected
        if id_hash in self.nodes:
            self.debug_print(f"connect_node: Already connected to {id_hash}")
            return True

        if can_connect:
            try:
                our_port = self.get_next_port()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind((self.host, our_port))
                sock.connect((host, port))
            except Exception as e:
                self.debug_print(
                    f"connect_node: could not initialize connection to {host}:{port} -> {e}"
                )
                return False

            self.debug_print(f"connect_node: connecting to {host}:{port}")
            message = pickle.dumps((None, self.role, self.id, self.rsa_pub_key))
            sock.send(message)
            return self.handshake(sock, (host, port), instigator=True)
        else:
            return False

    def get_validator_count(self):
        """Get number of listed validators on Smart Nodes"""
        num_validators = self.contract.functions.getValidatorCount().call()
        return num_validators

    def get_validator_info(self, validator_ind: int):
        """Get validator info from Smart Nodes"""
        validator_state = self.contract.functions.getValidatorInfo(validator_ind).call()

        # If validator was active at last state update, retrieve id and request connection info
        if validator_state:
            # Get validator information from smart contract
            return validator_state[0], validator_state[1]

        else:
            return None

    def bootstrap(self):
        """Bootstrap node to existing validators"""
        if self.off_chain_test is True:
            return

        n_validators = self.get_validator_count()
        sample_size = min(n_validators, 6)
        candidates = []

        # Connect to randomly selected validators
        while len(candidates) < sample_size and len(self.validators) < sample_size:
            # Random validator id
            validator_id = random.randrange(1, n_validators + 1)

            # Get key validator information from smart contract
            validator_contract_info = self.get_validator_info(validator_id)

            if validator_contract_info is not None:
                is_active, id_hash = validator_contract_info
                id_hash = id_hash.encode()
                validator_p2p_info = self.query_dht(id_hash)

                if validator_p2p_info is None:
                    self.delete(id_hash)
                    continue

                # Check to see if we are already connected
                already_connected = False
                for node in self.nodes.values():
                    if (
                        node.host == validator_p2p_info["host"]
                        and node.port == validator_p2p_info["port"]
                    ):
                        already_connected = True
                        break

                # Connect to the validator's node and exchange information
                # TODO what if we receive false connection info from validator: how to report?
                connected = self.connect_node(
                    id_hash, validator_p2p_info["host"], validator_p2p_info["port"]
                )

                if not connected:
                    self.delete(id_hash)
                    continue

                candidates.append(validator_id)

    def query_dht(
        self, key_hash: bytes, requester: bytes = None, ids_to_exclude: list = None
    ):
        """
        Retrieve stored value from DHT or query the closest node to a given key.
        * should be run in its own thread due to blocking RPC
        """
        self.debug_print(f"Querying DHT for {key_hash}")
        closest_node = None
        closest_distance = float("inf")

        # Find nearest node in our routing table
        for node_hash, node in self.routing_table.items():
            # Get XOR distance between keys
            distance = calculate_xor(key_hash, node_hash)

            if distance < closest_distance:
                if not ids_to_exclude or node_hash not in ids_to_exclude:
                    closest_node = (node_hash, node)
                    closest_distance = distance

        if requester is None:
            requester = self.rsa_key_hash

        if closest_node is not None:
            # The case where the stored value was not properly deleted (ie is None)
            if closest_node[1] is None:
                self.delete(closest_node[0])

                # Get another closest node
                return self.query_dht(key_hash, requester, ids_to_exclude)

            # The case where we have the stored value
            elif closest_node[0] == key_hash:
                # If the stored value is a node, send back its info
                if isinstance(closest_node[1], Connection):
                    return closest_node[1].stats

                # Else the value is a stored data structure (ie a job) and we can return it
                else:
                    return closest_node[1]

            # We don't have the stored value, and must route the request to the nearest node
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
        key_hash: bytes,
        node: Connection,
        requester: bytes = None,
        ids_to_exclude: list = None,
    ):
        """Query a specific node for a value"""
        if requester is None:
            requester = self.rsa_key_hash

        start_time = time.time()
        message = b"REQUEST-VALUE" + key_hash + requester
        self.send_to_node(node, message)

        # Logs what value were requesting and from what node
        self.store_request(node.node_id, key_hash)

        # Blocking wait to receive the data
        while key_hash not in self.routing_table:
            # TODO some better timeout management
            # Wait for 3 seconds and then find a new node to query
            if time.time() - start_time > 3:
                if ids_to_exclude is not None:
                    if len(ids_to_exclude) > 1:
                        return None
                    ids_to_exclude.append(node.node_id)
                else:
                    ids_to_exclude = [node.node_id]

                # Re route request to the next closest node
                self.requests[node.node_id].remove(key_hash)
                return self.query_dht(key_hash, requester, ids_to_exclude)

            if ids_to_exclude and len(ids_to_exclude) > 1:
                return None

        return self.routing_table[key_hash]

    def store_value(self, key: bytes, value: object, replicate: object = 0) -> object:
        """Store value in routing table and replicate if specified"""
        bucket_index = self.calculate_bucket_index(key)
        bucket = self.buckets[bucket_index]

        if not bucket.is_full():
            self.routing_table[key] = value
            bucket.add_node(self.routing_table[key])

        if 5 > replicate > 0:
            # TODO
            n_validators = self.get_validator_count()

            while replicate > 0:
                random_id = random.randrange(1, n_validators + 1)

                replicate -= 1
                pass

    def save_dht_state(self):
        """Serialize and save the DHT state to a file."""
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(self.routing_table, f)
            self.debug_print("DHT state saved successfully.")

        except Exception as e:
            self.debug_print(f"Error saving DHT state: {e}")

    def load_dht_state(self):
        """Load the DHT state from a file."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)

                self.routing_table = state
                self.debug_print("DHT state loaded successfully.")

            except Exception as e:
                self.debug_print(f"Error loading DHT state: {e}")

    def periodic_save_dht_state(self):
        """Periodically save the DHT state."""
        while not self.terminate_flag.is_set():
            self.save_dht_state()
            time.sleep(600)

    def request_store_value(self):
        pass

    def store_request(self, node_id: bytes, key_hash: bytes):
        """Stores a log of the request we have made to a node and for what value"""
        if node_id in self.requests:
            self.requests[node_id].append(key_hash)
        else:
            self.requests[node_id] = [key_hash]

    def delete(self, key: bytes):
        """
        Delete a key-value pair from the DHT.
        """

        if key in self.routing_table:
            val = self.routing_table[key]
            if isinstance(val, dict):
                if val["role"] == "U":
                    self.users.remove(key)
                elif val["role"] == "V":
                    self.validators.remove(key)
                elif val["role"] == "W":
                    self.validators.remove(key)

            del self.routing_table[key]
            self.debug_print(f"Key '{key}' deleted from DHT.")
        else:
            self.debug_print(f"Key '{key}' not found in DHT.")

    def update_routing_table(self):
        while not self.terminate_flag.is_set():
            for key, value in self.routing_table.items():
                if key in self.nodes:
                    pass
                elif key in self.jobs:
                    # TODO method / request to delete job after certain time or by request of the user.
                    #   Perhaps after a job is finished there is a delete request
                    pass
                else:
                    self.debug_print(f"Cleaning up item: {key} from routing table.")
                    self.delete(key)

            for key, node in self.nodes.items():
                if key not in self.routing_table:
                    self.debug_print(f"Adding: {key} to routing table.")
                    self.routing_table[key] = get_connection_info(node)

    def calculate_bucket_index(self, key: bytes):
        """
        Find the index of a bucket given the key
        """
        key_int = int(key, 16)
        bucket_index = key_int % len(self.buckets)

        return bucket_index

    def init_sock(self) -> None:
        """Initializes the main socket for handling incoming connections"""
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = self.get_next_port()
        self.port = port
        self.sock.bind((self.host, port))
        self.sock.settimeout(10)
        self.sock.listen(1)

    def init_upnp(self) -> None:
        """Enables UPnP on main socket to allow connections"""
        # if self.upnp:
        self.upnp = UPnP()
        self.upnp.discoverdelay = 10_000
        self.upnp.discover()
        self.upnp.selectigd()

    def add_port_mapping(self, external_port, internal_port):
        result = self.upnp.addportmapping(
            external_port, "TCP", self.upnp.lanaddr, internal_port, "SmartNode", ""
        )

        if result:
            self.debug_print(f"UPnP port forward successful on port {self.port}")
        else:
            self.debug_print("Failed to initialize UPnP.")

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
        """Makes sure we are not trying to connect to ourselves or a connected node"""
        # Check if trying to connect to self
        if host == self.host and port == self.port:
            self.debug_print("connect_with_node: cannot connect with yourself!")
            return False

        # Check if already connected
        for node in self.nodes.values():
            if node.host == host and node.port == port:
                self.debug_print(
                    f"connect_with_node: already connected with node: {node.node_id}"
                )
                return False

        return True

    def send_to_node(
        self, n: Connection, data: bytes, compression: bool = False
    ) -> None:
        """Send data to a connected node"""
        if n in self.nodes.values():
            n.send(data, compression=compression)
        else:
            self.debug_print("send_to_node: node not found!")

    def update_node_stats(
        self,
        node_hash: bytes,
        statistic_key: str,
        additional_context=None,
        decrement=False,
    ):
        """Updates node (connection) statistics, acts as an incrementer (default),
        sets the value if context is specified."""
        if additional_context is None:
            if node_hash in self.node_stats.keys():
                if decrement:
                    self.node_stats[node_hash][statistic_key] -= 1
                else:
                    self.node_stats[node_hash][statistic_key] += 1
            else:
                self.node_stats[node_hash] = {statistic_key: 1}
        else:
            if node_hash in self.node_stats.keys():
                self.node_stats[node_hash][statistic_key] = additional_context
            else:
                self.node_stats[node_hash] = {statistic_key: additional_context}

    def close_connection(self, n: Connection) -> None:
        n.stop()
        self.debug_print(f"node {n.node_id} disconnected")

    def handle_message(self, node: Connection, data) -> None:
        """Callback method to handles incoming data from connections"""
        self.debug_print(
            f"handle_message from {node.host}:{node.port} -> {data.__sizeof__()/1e6}MB"
        )
        self.handle_data(data, node)

    def ping_node(self, n: Connection):
        """Measure latency node latency"""
        n.pinged = time.time()
        self.send_to_node(n, b"PING")

    def close_connection_socket(
        self, n: socket.socket, additional_info: str = None
    ) -> None:
        message = "closing connection"
        if additional_info:
            message += f": {additional_info}"

        self.debug_print(message)
        n.close()

    def stop_upnp(self) -> None:
        """Shuts down UPnP on port"""
        if self.upnp:
            self.upnp.deleteportmapping(self.port, "TCP")
            self.debug_print(f"stop_upnp: UPnP removed for port {self.port}")

    def stop(self) -> None:
        """Shut down node and all associated connections/threads"""
        self.debug_print(f"Node stopping.")
        self.terminate_flag.set()
        if self.role == b"U":
            self.endpoint_thread.join()
        self.stop_upnp()

    # Methods to interact with Flask endpoints
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
                    port += 1

                except Exception as e:
                    raise e
            else:
                port += 1
