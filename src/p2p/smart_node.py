from src.cryptography.substrate import load_substrate_keypair
from src.cryptography.rsa import *
from src.p2p.connection import Connection
from src.p2p.node import Node

from web3 import Web3

import threading
import hashlib
import random
import socket
import pickle
import time
import json
import os


RPC = "http://127.0.0.1:7545"
with open("./src/assets/SmartNodes.json", "r") as f:
    METADATA = json.load(f)

ABI = METADATA["abi"]
CONTRACT_ADDRESS = "0xb89513D0F115a1c83f55513531BA2600cA087cEF"


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


def get_connection_info(node):
    info = {
        "host": node.host,
        "port": node.parent_port,
        "role": node.role,
        "id": node.node_id,
    }
    return info


class Bucket:
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


class SmartNode(Node):
    """
    TODO:
    - confirm workers public key with smart contract ID
    """

    def __init__(
        self,
        host: str,
        port: int,
        public_key: str,
        url: str = RPC,
        contract: str = CONTRACT_ADDRESS,
        debug: bool = False,
        max_connections: int = 0,
        dht_callback=None,
    ):
        super(SmartNode, self).__init__(
            host, port, debug, max_connections, self.stream_data
        )
        # DHT Parameters
        self.replication_factor = 3
        self.bucket_size = 2
        self.routing_table = {}
        self.buckets = [Bucket(d, self.bucket_size) for d in range(256)]
        self.nodes = {}
        self.dht_callback = dht_callback

        # Smart contract parameters
        self.chain = Web3(Web3.HTTPProvider(url))
        self.contract_address = Web3.to_checksum_address(contract)
        self.contract = None
        self.key_hash = hashlib.sha256(self.rsa_pub_key).hexdigest()

        # Grab the SmartNode contract
        try:
            self.contract = self.chain.eth.contract(
                address=self.contract_address, abi=ABI
            )

        except Exception as e:
            self.debug_print(f"Could not retrieve smart contract: {e}")
            self.terminate_flag.set()

        self.updater_flag = threading.Event()

    def stream_data(self, data: bytes, node):
        """
        Handle incoming data streams from connected nodes and process requests.
        """
        try:
            # The case where we load via downloaded pickle file (potential security threat?)
            if b"DONE STREAM" == data[:11]:
                file_name = f"streamed_data_{node.host}_{node.port}"

                with open(file_name, "rb") as f:
                    streamed_bytes = f.read()

                self.stream_data(streamed_bytes, node)

                os.remove(file_name)

            elif b"REQUESTP" == data:
                self.debug_print(f"RECEIVED PEER REQUEST")
                self.handle_peer_request(node)

            elif b"RESPONSEP" == data[:9]:
                self.debug_print(f"RECEIVED PEERS")

                pickled = pickle.loads(data[9:])
                # node_id, peer_ids = pickled

                for host, port in pickled:
                    if self.connect_with_node(host, port, our_node_id=self.key_hash):
                        new_peer = next(
                            (
                                node
                                for node in self.connections
                                if node.host == host and node.port == port
                            ),
                            None,
                        )

            elif b"STORE" == data[:5]:
                # Store the key-value pair in the DHT
                key, value = pickle.loads(data[5:])
                self.store_key_value_pair(key, value)

            elif b"ROUTEREQ" == data[:8]:
                # Retrieve the value associated with the key from the DHT
                self.debug_print(f"RECEIVED ROUTE REQUEST")
                key = data[8:]
                value = self.query_routing_table(key)
                data = pickle.dumps([key, value])
                data = b"ROUTEREP" + data

                # Send the value back to the requesting node
                self.send_to_node(node, data)

            elif b"ROUTEREP" == data[:8]:
                self.debug_print(f"RECEIVED ROUTE RESPONSE")
                key, value = pickle.loads(data[8:])

                if value is not None:
                    self.routing_table[key] = value

            elif b"DELETE" == data[:6]:
                # Delete the key-value pair from the DHT
                key = pickle.loads(data[6:])
                self.delete(key)

            else:
                return False

            return True

        except Exception as e:
            self.debug_print(f"Error handling stream data: {e}")

    def run(self):
        # Thread for handling incoming connections
        listener = threading.Thread(target=self.listen, daemon=True)
        listener.start()

        while not self.terminate_flag.is_set():
            self.reconnect_nodes()

        print("Node stopping...")
        for node in self.connections:
            node.stop()

        time.sleep(1)

        for node in self.connections:
            node.join()

        self.sock.settimeout(None)
        self.sock.close()
        print("Node stopped")

    def handshake(self, connection, client_address):
        """
        Validates incoming connection's keys via a random number swap, along with SC
            verification of connecting user
        """

        connected_node_id = connection.recv(4096)
        role, connected_node_id = connected_node_id[0:1], connected_node_id[1:]

        # Generate random number to confirm with incoming node
        randn = str(random.random())
        port = random.randint(5000, 65000)

        message = f"{randn},{port},{self.key_hash}"

        # All nodes should have a valid RSA ID
        if authenticate_public_key(connected_node_id) is True:
            start_time = time.time()

            # Role-specific smart contract and DHT credential/reputation validation
            if role == b"V":
                # Further confirmation of validator key via smart contract
                try:
                    verified_public_key = self.contract.functions.validatorIdByHash(
                        hashlib.sha256(connected_node_id).hexdigest()
                    ).call()

                    if verified_public_key < 1:
                        connection.close()
                        return

                except Exception as e:
                    self.debug_print(f"Contract query error: {e}")

            elif role == b"U":
                # Some check of the DHT for any user reputation / blacklisted users
                pass

            elif role == b"W":
                # Some check of the DHT for any user reputation / blacklisted users
                pass

            # Encrypt random number with node's key to confirm identity
            proof_response = encrypt(message.encode(), self.port, connected_node_id)
            message = proof_response + b"," + self.role + self.rsa_pub_key
            connection.send(message)

            # Await response
            response = connection.recv(4096)
            latency = time.time() - start_time
            response = decrypt(response, self.port)
            response, parent_port, node_id, verification = response.split(b",")

            connection.send(verification)

            if response.decode() == randn:
                thread_client = self.create_connection(
                    connection,
                    client_address[0],
                    client_address[1],
                    node_id,
                    int(parent_port),
                    role,
                )
                thread_client.start()
                thread_client.latency = latency

                for node in self.connections:
                    if (
                        node.host == client_address[0]
                        and node.port == port
                        or node.parent_port == port
                    ):
                        self.debug_print(
                            f"connect_with_node: already connected with node: {node.node_id}"
                        )
                        thread_client.stop()
                        break

                if not thread_client.terminate_flag.is_set():
                    self.nodes[node_id] = thread_client
                    self.connections.append(self.nodes[node_id])

                    self.connect_dht_node(
                        thread_client.host,
                        thread_client.parent_port,
                        role,
                        connected=True,
                    )
            else:
                self.debug_print(
                    f"SmartNode: Connection refused, validator not registered!"
                )
                connection.close()

        else:
            self.debug_print("SmartNode: connection refused, invalid proof!")
            connection.close()

    def listen(self):
        """
        Listen for incoming connections on main port and confirm via custom handshake
        """
        while not self.terminate_flag.is_set():
            # Accept validation connections from registered nodes
            try:
                # Unpack connection info
                connection, client_address = self.sock.accept()

                # Attempt SC-secured connection if we can handle more
                if (
                    self.max_connections == 0
                    or len(self.connections) < self.max_connections
                ):
                    self.handshake(connection, client_address)

                else:
                    self.debug_print(
                        "node: Connection refused: Max connections reached!"
                    )
                    connection.close()

            except socket.timeout:
                self.debug_print("node: Connection timeout!")

            except Exception as e:
                print(str(e))

            self.reconnect_nodes()
            time.sleep(0.1)

    def connect_dht_node(
        self, host: str, port: int, reconnect: bool = False, connected=None
    ) -> bool:
        """
        Connect to a DHT node and exchange information to identify its node ID.
        """
        # Check if trying to connect to self
        if host == self.host and port == self.port:
            self.debug_print("connect_with_node: cannot connect with yourself!")
            return False

        if connected is None:
            # Check if already connected
            for node in self.connections:
                if node.host == host and node.port == port or node.parent_port == port:
                    self.debug_print(
                        f"connect_with_node: already connected with node: {node.node_id}"
                    )
                    return True

            try:
                # Initialize a socket for potential connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((host, port))
                self.debug_print(f"connecting to {host} port {port}")

                start_time = time.time()

                # ID exchange
                sock.send(self.role + self.rsa_pub_key)
                verification = sock.recv(4096)
                latency = time.time() - start_time
                verification, their_pub_key = verification.split(b",")
                their_role, their_pub_key = their_pub_key[0:1], their_pub_key[1:]
                proof = decrypt(verification, self.port)
                proof, new_port, node_id = proof.split(b",")

                # Now we must verify the other nodes credentials
                randn = str(random.random())
                message = (
                    proof
                    + b","
                    + f"{self.port}".encode()
                    + b","
                    + self.key_hash.encode()
                    + b","
                    + randn.encode()
                )
                encrypted_message = encrypt(message, port, their_pub_key)
                sock.send(encrypted_message)
                response = sock.recv(4096)

                if randn == response.decode():
                    # Form connection
                    thread_client = self.create_connection(
                        sock, host, int(new_port), node_id, port, role=their_role
                    )
                    thread_client.start()
                    thread_client.latency = latency

                    self.nodes[node_id] = thread_client
                    self.connections.append(self.nodes[node_id])

                    # If reconnection to this host is required, add to the list
                    if reconnect:
                        self.debug_print(
                            f"connect_with_node: reconnection check enabled on {host}:{port}"
                        )
                        self.reconnect.append({"host": host, "port": port, "tries": 0})

                        return False

                    connected = True

                else:
                    sock.close()
                    self.debug_print(
                        f"SmartNode: Connection refused, invalid proof!: {node_id}"
                    )

            except Exception as error:
                self.debug_print(
                    f"connect_with_node: could not connect with node. ({error})"
                )
                return False

        if connected:
            # Find the node we want to store
            node = None
            for n in self.connections:
                if (n.host, n.parent_port) == (host, port):
                    node = n
                    break

            node_info = get_connection_info(node)
            self.store_key_value_pair(node.node_id, node_info)
            return True

        return False

    # def get_jobs(self):
    #     # Confirm job details with smart contract, receive initial details from a node?
    #     # self.chain.query("")
    #     pass
    #
    # def get_seed_workers(self, job_id):
    #     try:
    #         job_details = self.chain.query(
    #             module="Contracts",
    #             storage_function="GetSeedWorkers",
    #             params=[self.contract_address, job_id]
    #         )
    #
    #         return job_details
    #
    #     except SubstrateRequestException as e:
    #         self.debug_print(f"Failed to get job details: {e}")
    #         return None
    #
    # def get_user_info(self, user_address):
    #
    #     try:
    #         user_info = self.chain.compose_call(
    #             call_module="",
    #             call_function="",
    #             call_params={}
    #         )
    #     pass

    def handle_peer_request(self, requesting_node):
        """
        Handle requests from other nodes to provide a list of neighboring peers.
        """
        peers = [(node.host, node.parent_port) for node in self.connections]
        message = b"RESPONSEP" + pickle.dumps(peers)
        self.send_to_node(requesting_node, message)

    def bootstrap(self):
        num_validators = self.contract.functions.getValidatorIdCounter().call() - 1
        sample_size = min(num_validators, 10)  # Adjust sample size as needed

        # Randomly select sample_size validators
        random_sample = random.sample(range(1, num_validators + 1), sample_size)
        random_sample = [1, 2, 3]

        for validatorId in random_sample:

            # Get validator information from smart contract
            _, address, id_hash, reputation, active = (
                self.contract.functions.validators(validatorId).call()
            )

            node_info = self.query_routing_table(id_hash.encode())
            # host, port = connection_info["host"], connection_info["port"]

            if node_info is None:
                self.delete(id_hash)
                continue

            # Connect to the validator's node and exchange information
            connected = self.connect_dht_node(node_info["host"], node_info["port"])

            # Check to see if connected, if not we can try another random node
            if not connected:
                self.delete(id_hash)

    def query_routing_table(self, key_hash, ids_to_exclude=[]):
        """
        Get the node responsible for, or closest to a given key.
        """
        closest_node = None
        closest_distance = float("inf")

        # Find nearest node in our local routing table
        for node_hash, node in self.routing_table.items():
            distance = calculate_xor(key_hash, node_hash)
            if distance < closest_distance:
                if not ids_to_exclude or node_hash not in ids_to_exclude:
                    closest_node = (node_hash, node)
                    closest_distance = distance

        # In the case we have the target query value
        if closest_node[0] == key_hash:
            if closest_node[1] is None:
                self.delete(closest_node[0])
                return None
            return self.routing_table[closest_node[0]]

        # If we could not retrieve the stored value, route request to nearest node
        else:
            if closest_node[0] in self.nodes:
                start_time = time.time()
                node = self.nodes[closest_node[0]]
                self.request_value(key_hash, node)

                while key_hash not in self.routing_table.keys():
                    if (
                        time.time() - start_time > 1000
                    ):  # Some arbitrary timeout time for now
                        if len(ids_to_exclude) >= 1:
                            return None

                        ids_to_exclude.append(closest_node[0])
                        return self.query_routing_table(
                            key_hash,
                            ids_to_exclude,
                        )

                return self.routing_table[key_hash]
            else:
                return None

    def request_value(self, key: bytes, node: Connection):
        data = b"ROUTEREQ" + key
        self.send_to_node(node, data)

    def store_key_value_pair(self, key: bytes, value):
        key_int = int(key, 16)
        bucket_index = self.calculate_bucket_index(key_int)
        bucket = self.buckets[bucket_index]

        if not bucket.is_full():
            self.routing_table[key] = value
            bucket.add_node(self.routing_table[key])
            return True

        else:
            # Pass along to another node (x replication factor)
            target_node = self.query_routing_table(key)
            self.store_key_value_pair_with_acknowledgment(key, value, target_node)

        # Replicate the data to the next closest nodes
        # for i in range(self.replication_factor):
        #     next_node = self.get_next_node(key, node)
        #     if next_node:
        #         next_node.store(key, value)

    def store_key_value_pair_with_acknowledgment(self, key, value, node):
        pass

    def forward_to_other_node(self, key, value):
        target_node = self.query_routing_table(hash_key(key))
        pickled = pickle.dumps((key, value))
        self.send_to_node(target_node, b"STORE" + pickled)

    def calculate_bucket_index(self, key_int):
        """
        Find the index of a bucket given the key
        """
        bucket_index = key_int % len(self.buckets)

        return bucket_index

    def delete(self, key):
        """
        Delete a key-value pair from the DHT.
        """
        if isinstance(key, str):
            key = key.encode()

        if key in self.routing_table:
            del self.routing_table[key]
            self.debug_print(f"Key '{key}' deleted from DHT.")
        else:
            self.debug_print(f"Key '{key}' not found in DHT.")
