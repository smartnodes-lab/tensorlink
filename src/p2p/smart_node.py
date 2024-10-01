from src.crypto.rsa import *
from src.p2p.connection import Connection

from logging.handlers import TimedRotatingFileHandler
from miniupnpc import UPnP
from web3 import Web3
import ipaddress
import threading
import requests
import logging
import hashlib
import pickle
import random
import socket
import json
import time
import os


# Grab smart contract information
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, '..', 'config', 'SmartnodesCore.json')
ms_config_path = os.path.join(base_dir, '..', 'config', 'SmartnodesMultiSig.json')

with open(os.path.join(base_dir, "..", "config", "config.json"), "r") as f:
    config = json.load(f)
    CHAIN_URL = config["api"]["chain-url"]
    CONTRACT = config["api"]["core"]
    MULTI_SIG_CONTRACT = config["api"]["multi-sig"]

with open(config_path, "r") as f:
    METADATA = json.load(f)
ABI = METADATA["abi"]

with open(ms_config_path, "r") as f:
    MS_METADATA = json.load(f)
MULTI_SIG_ABI = MS_METADATA["abi"]

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


def get_connection_info(node, main_port=None):
    """Connection info for routing table storage"""
    node_host = node.host

    if is_private_ip(node_host):
        node_host = get_public_ip()

    info = {
        "host": node.host,
        "port": node.port if not main_port else main_port,
        "role": node.role,
        "id": node.node_id,
        "reputation": node.reputation,
    }

    return info


def log_entry(node, metadata):
    entry = {"timestamp": time.ctime(), "nodes": node.node_id, "metadata": metadata}
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
    A P2P nodes secured by RSA encryption and smart contract validation for the Smartnodes ecosystem.
    Combines smart contract queries and a kademlia-like DHT implementation for data storage and access.
    """

    def __init__(
        self,
        debug: bool = False,
        max_connections: int = 0,
        upnp: bool = True,
        off_chain_test: bool = False,
        debug_colour=None
    ):
        super(SmartNode, self).__init__()

        # Node info
        self.terminate_flag = threading.Event()
        self.connection_listener = None

        # Connection info
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

        # Connection Settings
        self.upnp = None
        self.nodes = {}  # nodes-hash: Connection
        self.node_stats = (
            {}
        )  # nodes-hash: {message counts & other key stats to keep track of}

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
        self.rsa_pub_key = None
        self.rsa_key_hash = None
        self.role = b""
        self.id = 0

        # Stores key of stored values
        self.validators = []
        self.workers = []
        self.users = []
        self.jobs = []

        self.sno_events = {name: Web3.keccak(text=sig).hex() for name, sig in SNO_EVENT_SIGNATURES.items()}
        self.off_chain_test = off_chain_test

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
                self.debug_print(f"Could not retrieve contract: {e}")
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
                    f"streamed_data_{node.host}_{node.port}_{self.host}_{self.port}"
                )

                if b"MODULE" in data[11:]:
                    streamed_bytes = data[11:]
                    os.rename(file_name, data[17:].decode())

                else:
                    with open(file_name, "rb") as file:
                        streamed_bytes = file.read()
                        os.remove(file_name)

                self.handle_data(streamed_bytes, node)

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
                    f"handle_data: nodes ({node.host}:{node.port}) returned value."
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
                    f"handle_data: nodes ({node.host}:{node.port}) requested value."
                )
                validator_info = None
                value_hash = None

                if len(data) < 141:
                    # TODO not enough data received!
                    pass
                else:
                    value_hash = data[13:77]
                    requester = data[77:141]

                    # Get nodes info
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
                reset_colour = "\033[0m"
                print(f"{self.debug_colour}{self.host}:{self.port} -> {message}{reset_colour}")

    """Methods for DHT Query and Storage"""
    def query_dht(
        self, key_hash: bytes, requester: bytes = None, ids_to_exclude: list = None
    ):
        """
        Retrieve stored value from DHT or query the closest nodes to a given key.
        * should be run in its own thread due to blocking RPC
        """
        self.debug_print(f"Querying DHT for {key_hash}")
        closest_node = None
        closest_distance = float("inf")

        # Find nearest nodes in our routing table
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

                # Get another closest nodes
                return self.query_dht(key_hash, requester, ids_to_exclude)

            # The case where we have the stored value
            elif closest_node[0] == key_hash:
                # If the stored value is a nodes, send back its info
                if isinstance(closest_node[1], Connection):
                    return closest_node[1].stats

                # Else the value is a stored data structure (ie a job) and we can return it
                else:
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
        key_hash: bytes,
        node: Connection,
        requester: bytes = None,
        ids_to_exclude: list = None,
    ):
        """Query a specific nodes for a value"""
        if requester is None:
            requester = self.rsa_key_hash

        start_time = time.time()
        message = b"REQUEST-VALUE" + key_hash + requester
        self.send_to_node(node, message)

        # Logs what value were requesting and from what nodes
        self.store_request(node.node_id, key_hash)

        # Blocking wait to receive the data
        while key_hash not in self.routing_table:
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

    def store_request(self, node_id: bytes, key: bytes):
        """Stores a log of the request we have made to a nodes and for what value"""
        if node_id in self.nodes.keys():
            if node_id in self.requests:
                self.requests[node_id].append(key)
            else:
                self.requests[node_id] = [key]

    def remove_request(self, node_id: bytes, key: bytes):
        if node_id in self.requests:
            self.requests[node_id].remove(key)

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

    def clean_up_dht(self):
        # TODO Remove old nodes, old jobs, store some things in files
        pass

    """Peer-to-peer methods"""
    def listen(self):
        """Listen for incoming connections and initialize custom handshake"""
        while not self.terminate_flag.is_set():
            try:
                # Unpack nodes info
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
        """Validates incoming nodes's keys via a random number swap, along with SC verification of connecting user"""
        id_info = connection.recv(4096)
        _, role, id_no, connected_node_id = pickle.loads(id_info)
        node_id_hash = hashlib.sha256(connected_node_id).hexdigest().encode()

        # Query nodes info/history from dht for reputation if we are a validator
        if self.role == b"V" or b"V2":
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
                try:

                    if role == b"V":
                        # Query contract for users key hash
                        is_active, pub_key_hash, wallet_address = self.contract.functions.getValidatorInfo(
                            id_no
                        ).call()

                        # If validator was not found
                        if not is_active or node_id_hash != pub_key_hash:
                            self.update_node_stats(node_id_hash, "GHOST")
                            # TODO: potentially some form of reporting mechanism via ip and port
                            raise f"listed on Smart Nodes!: {connected_node_id}"

                    elif role == b"U":
                        # TODO: user handling, to be done once users are required to register (post-alpha)
                        pass

                    elif role == b"W":
                        # TODO: worker handling
                        pass

                    else:
                        # TODO: potentially some form of reporting mechanism via ip and port
                        raise f"listen: connection refused, invalid role: {node_address}"

                except Exception as e:
                    self.close_connection_socket(
                        connection, f"handshake: contract query error: {e}"
                    )

            # Random number swap to confirm the nodes RSA key
            rand_n = random.random()
            encrypted_number = encrypt(str(rand_n).encode(), self.role, connected_node_id)

            # Encrypt random number with nodes's key to confirm their identity
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
                    new_port, rand_n_proof = pickle.loads(response)
                    rand_n_proof = float(rand_n_proof)
                    main_port = node_address[1]

                    # Somehow switch the connection to the nodes new port
                    connection.close()
                    self.debug_print(f"Switching connection to the new port: {new_port}")

                    # Select a new port for the node to use if we are not the instigator
                    our_new_port = self.get_next_port()
                    self.debug_print(f"Selected next port: {our_new_port} for new connection")
                    self.add_port_mapping(our_new_port, our_new_port)

                    # Establish a new connection to the node on the provided port
                    new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    new_sock.bind((self.host, our_new_port))
                    new_sock.connect((node_address[0], new_port))

                    connection = new_sock

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

                # Select a new port for the node to use if we are not the instigator
                our_new_port = self.get_next_port()
                self.debug_print(f"Selected next port: {our_new_port} for new connection")
                self.add_port_mapping(our_new_port, our_new_port)

                # Send the new port and proof of random number
                response = pickle.dumps((our_new_port, verification))
                connection.send(response)

                # Close the current connection and listen on the new port
                connection.close()
                self.debug_print(f"Listening for the instigator on the new port: {our_new_port}")

                # Create a new socket and bind to the selected port
                new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                new_sock.bind((self.host, our_new_port))
                new_sock.listen(2)

                # Accept the incoming connection on the new port
                connection, new_node_address = new_sock.accept()

            # If the nodes has confirmed his identity
            if rand_n_proof == rand_n:
                # Check to see if we are already connected
                for node in self.nodes.values():
                    if node.host == node_address[0] and node.port == our_new_port:
                        self.debug_print(
                            f"connect_with_node: already connected with nodes: {node.node_id}"
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

                # Finally connect to the nodes
                if not thread_client.terminate_flag.is_set():
                    self.debug_print(
                        f"Connected to nodes: {thread_client.host}:{thread_client.port}"
                    )
                    self.nodes[node_id_hash] = thread_client

                    # Check if nodes has history on the network
                    node_info = self.query_dht(node_id_hash)

                    if node_info is None:
                        # New nodes, broadcast info to others
                        node_info = get_connection_info(thread_client, main_port if not instigator else None)

                    elif node_info["reputation"] < 50:
                        raise "Connection rejected: Poor reputation."

                    self.store_value(node_id_hash, node_info)

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
        Connect to a nodes and exchange information to confirm its role in the Smart Nodes network.
        """
        can_connect = self.can_connect(host, port)

        # Check that we are not already connected
        if id_hash in self.nodes:
            self.debug_print(f"connect_node: Already connected to {id_hash}")
            return True

        if can_connect:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                time.sleep(0.1)
                our_port = self.get_next_port()
                self.add_port_mapping(our_port, our_port)
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

    def bootstrap(self):
        """Bootstrap nodes to existing validators"""
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

                # Connect to the validator's nodes and exchange information
                # TODO what if we receive false connection info from validator: how to report?
                connected = self.connect_node(
                    id_hash, validator_p2p_info["host"], validator_p2p_info["port"]
                )

                if not connected:
                    self.delete(id_hash)
                    continue

                candidates.append(validator_id)

    def init_sock(self) -> None:
        """Initializes the main socket for handling incoming connections"""
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = self.get_next_port()
        self.port = port
        self.add_port_mapping(port, port)
        self.sock.bind((self.host, port))
        self.sock.settimeout(5)
        self.sock.listen(1)

    def init_upnp(self) -> None:
        """Enables UPnP on main socket to allow connections"""
        self.upnp = UPnP()
        self.upnp.discoverdelay = 2_000
        self.upnp.discover()
        self.upnp.selectigd()
        self.add_port_mapping(self.port, self.port)

    def add_port_mapping(self, external_port, internal_port):
        if self.upnp:
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
        """Makes sure we are not trying to connect to ourselves or a connected nodes"""
        # Check if trying to connect to self
        if host == self.host and port == self.port:
            self.debug_print("connect_with_node: cannot connect with yourself!")
            return False

        # Check if already connected
        for node in self.nodes.values():
            if node.host == host and node.port == port:
                self.debug_print(
                    f"connect_with_node: already connected with nodes: {node.node_id}"
                )
                return False

        return True

    def send_to_node(
        self, n: Connection, data: bytes, compression: bool = False
    ) -> None:
        """Send data to a connected nodes"""
        if n in self.nodes.values():
            n.send(data, compression=compression)
        else:
            self.debug_print("send_to_node: nodes not found!")

    def send_to_node_from_file(self, n: Connection, file, tag):
        if n in self.nodes.values():
            n.send_from_file(file, tag)
        else:
            self.debug_print("send_to_node: nodes not found!")

    def update_node_stats(
        self,
        node_hash: bytes,
        statistic_key: str,
        additional_context=None,
        decrement=False,
    ):
        """Updates nodes (connection) statistics, acts as an incrementer (default),
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
        self.debug_print(f"nodes {n.node_id} disconnected")

    def handle_message(self, node: Connection, data) -> None:
        """Callback method to handles incoming data from connections"""
        self.debug_print(
            f"handle_message from {node.host}:{node.port} -> {data.__sizeof__()/1e6}MB"
        )
        self.handle_data(data, node)

    def ping_node(self, n: Connection):
        """Measure latency nodes latency"""
        n.pinged = time.time()
        self.send_to_node(n, b"PING")

    def run(self):
        self.connection_listener = threading.Thread(target=self.listen, daemon=True)
        self.connection_listener.start()

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
        """Shut down nodes and all associated connections/threads"""
        self.debug_print(f"Node stopping...")
        self.terminate_flag.set()
        self.connection_listener.join()

        for node in self.nodes.values():
            node.stop()

        for node in self.nodes.values():
            node.join()

        self.sock.settimeout(None)
        self.sock.close()

        if self.role == b"U":
            self.endpoint_thread.join()

        self.stop_upnp()

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
                    port += 1

                except Exception as e:
                    raise e
            else:
                port += 1

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

    """Methods for Smart Contract Interactions"""
