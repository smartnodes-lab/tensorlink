from src.cryptography.rsa import *
from src.cryptography.substrate import load_substrate_keypair
from src.p2p.node import Node

from substrateinterface import SubstrateInterface
from substrateinterface.contracts import ContractCode, ContractInstance, ContractMetadata
from substrateinterface.exceptions import SubstrateRequestException

import threading
import hashlib
import random
import socket
import pickle
import time
import os


METADATA = "./src/assets/smartnodes.json"
CONTRACT = "5D37KYdd3Ptd8CfjKxtN3rGqU6oZQQeiGfTLfF2VGrTQJfyN"


def hash_key(key: bytes):
    """
    Hashes the key to determine its position in the keyspace.
    """
    return int(hashlib.sha256(key).hexdigest(), 16)


def calculate_xor(key_hash, node_id):
    """
    Calculate the XOR distance between a key and a node ID.
    """
    return key_hash ^ int(node_id, 16)


class Bucket:
    def __init__(self, max_values):
        self.values = []
        self.max_values = max_values

    def is_full(self):
        return len(self.values) >= self.max_values

    def add_node(self, value):
        if not self.is_full():
            self.values.append(value)

    def remove_node(self, value):
        if value in self.values:
            self.values.remove(value)


class SmartDHTNode(Node):
    """
    TODO:
    - confirm workers public key with smart contract ID
    """
    def __init__(self, host: str, port: int, public_key: str, url: str = "wss://ws.test.azero.dev",
                 contract: str = CONTRACT, debug: bool = False,
                 max_connections: int = 0, callback=None):
        super(SmartDHTNode, self).__init__(host, port, debug, max_connections, self.stream_data)

        # Smart contract parameters
        self.chain = SubstrateInterface(url=url)
        self.keypair = load_substrate_keypair(public_key, "      ")
        self.contract_address = contract
        self.contract = None

        # Grab the SmartNode contract
        contract_info = self.chain.query("Contracts", "ContractInfoOf", [self.contract_address])
        if contract_info.value:
            self.contract = ContractInstance.create_from_address(
                substrate=self.chain,
                contract_address=self.contract_address,
                metadata_file=METADATA
            )

        else:
            self.debug_print("Could not retrieve smart contract.")
            self.terminate_flag.set()

        # DHT Parameters
        self.replication_factor = 3
        self.bucket_size = 2
        self.routing_table = {}
        self.buckets = [Bucket(self.bucket_size) for _ in range(256)]

        # self.key_hash = hashlib.sha256(self.rsa_pub_key).hexdigest()
        self.key_hash = hashlib.sha256(bytes(random.randint(0, 10000))).hexdigest()
        self.updater_flag = threading.Event()

    def stream_data(self, data: bytes, node):
        """
        Handle incoming data streams from connected nodes and process requests.
        """
        try:

            # The case where we load via downloaded pickle file (potential security threat)
            if b"DONE STREAM" == data[:11]:
                file_name = f"streamed_data_{node.host}_{node.port}"

                with open(file_name, "rb") as f:
                    streamed_bytes = f.read()

                self.stream_data(streamed_bytes, node)

                os.remove(file_name)

            elif b"STORE" == data[:5]:
                # Store the key-value pair in the DHT
                key, value = pickle.loads(data[5:])
                self.store_key_value_pair(key, value)

            elif b"RETRIEVE" == data[:8]:
                # Retrieve the value associated with the key from the DHT
                key = pickle.loads(data[8:])
                value = self.retrieve(key)
                # Send the value back to the requesting node
                self.send_to_node(node, pickle.dumps(value))

            elif b"DELETE" == data[:6]:
                # Delete the key-value pair from the DHT
                key = pickle.loads(data[6:])
                self.delete(key)

            elif b"REQUESTS" == data:
                self.debug_print(f"RECEIVED STATS REQUEST")
                self.handle_statistics_request(node)

            elif b"REQUESTP" == data:
                self.debug_print(f"RECEIVED PEER REQUEST")
                self.handle_peer_request(node)

            elif b"RESPONSES" == data[:9]:
                self.debug_print(f"RECEIVED NODE STATS")

                pickled = pickle.loads(data[9:])
                node_id, stats = pickled
                stats["connection"] = node
                self.nodes[node_id] = stats

            elif b"RESPONSEP" == data[:9]:
                self.debug_print(f"RECEIVED PEERS")

                pickled = pickle.loads(data[9:])
                # node_id, peer_ids = pickled

                for host, port in pickled:
                    if self.connect_with_node(host, port, our_node_id=self.key_hash):
                        new_peer = next((node for node in self.connections if node.host == host and node.port == port), None)

            # Add more data types and their handling logic as needed

        except Exception as e:
            self.debug_print(f"Error handling stream data: {e}")

    def handshake(self, connection, client_address):
        """
        Validates incoming connection's keys via a random number swap, along with SC
            verification of connecting user
        """
        connected_node_id = connection.recv(4096)

        # Generate random number to confirm with incoming node
        randn = str(random.random())
        port = random.randint(5000, 65000)
        message = f"{randn},{port},{self.key_hash}"

        # Authenticate incoming node's id is valid key
        if authenticate_public_key(connected_node_id) is True:
            # # Further confirmation of user key via smart contract
            # try:
            #     verified_public_key = self.contract.read(
            #         keypair=self.keypair, method="check_worker", args={"pub_key": connected_node_id}
            #     )
            #
            #     is_verified = verified_public_key.contract_result_data.value["Ok"]
            #
            #     if is_verified:
            id_bytes = connected_node_id
            start_time = time.time()

            # Encrypt random number with node's key to confirm identity
            connection.send(
                encrypt(message.encode(), id_bytes)
            )

            # Await response
            response = connection.recv(4096)
            latency = time.time() - start_time
            response, parent_port, node_id = response.split(b",")

            if response.decode() == randn:
                thread_client = self.create_connection(connection, client_address[0], client_address[1],
                                                       node_id, int(parent_port))
                thread_client.start()
                thread_client.latency = latency

                for node in self.connections:
                    if node.host == client_address[0] and node.port == port or node.parent_port == port:
                        self.debug_print(
                            f"connect_with_node: already connected with node: {node.node_id}")
                        thread_client.stop()
                        break

                if not thread_client.terminate_flag.is_set():
                    self.inbound.append(thread_client)
                    self.connect_dht_node(thread_client.host, thread_client.parent_port, connected=True)

            else:
                self.debug_print("node: connection refused, invalid ID proof!")
                connection.close()

            #     else:
            #         self.debug_print("User not listed on contract.")
            #
            # except SubstrateRequestException as e:
            #     self.debug_print(f"Failed to verify user public key: {e}")

        else:
            self.debug_print("node: connection refused, invalid ID proof!")
            connection.close()

    def listen(self):
        """
        Listen for incoming connections and confirm via custom handshake
        """
        while not self.terminate_flag.is_set():
            # Accept validation connections from registered nodes
            try:
                # Unpack connection info
                connection, client_address = self.sock.accept()

                # Attempt SC-secured connection if we can handle more
                if self.max_connections == 0 or len(self.connections) < self.max_connections:
                    self.handshake(connection, client_address)

                else:
                    self.debug_print(
                        "node: Connection refused: Max connections reached!")
                    connection.close()

            except socket.timeout:
                self.debug_print('node: Connection timeout!')

            except Exception as e:
                print(str(e))

            time.sleep(0.1)

    def run(self):
        # Thread for handling incoming connections
        listener = threading.Thread(target=self.listen, daemon=True)
        listener.start()

        # Thread for periodic worker statistics updates
        stats_updater = threading.Thread(target=self.update_worker_stats, daemon=True)
        stats_updater.start()

        # Main worker loop
        while not self.terminate_flag.is_set():
            self.reconnect_nodes()
            time.sleep(1)

        print("Node stopping...")
        for node in self.connections:
            node.stop()

        time.sleep(1)

        for node in self.connections:
            node.join()

        self.sock.settimeout(None)
        self.sock.close()
        print("Node stopped")

    def bootstrap(self, seeds=None):
        """
        Connect to initial set of validator nodes on the network. Select random set
         of validators or workers from the smart contract if seeds=None.
        """
        if seeds is None:
            self.c

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

    def calculate_bucket_index(self, key):
        key_hash = hashlib.sha256(str(key).encode()).hexdigest()
        hash_integer = int(key_hash, 16)
        bucket_index = hash_integer % len(self.buckets)
        return bucket_index

    def query_routing_table(self, key_hash):
        """
        Get the node responsible for, or closest to a given key.
        """
        closest_node = None
        closest_distance = float("inf")
        for node in self.connections:
            distance = calculate_xor(key_hash, node.node_id)
            if distance < closest_distance:
                closest_node = node
                closest_distance = distance

        return closest_node

    def connect_dht_node(self, host: str, port: int, reconnect: bool = False, connected=None) -> bool:
        """
        Connect to a DHT node and exchange information to identify its node ID.
        """
        if connected is None:
            connected = self.connect_with_node(host, port, reconnect, self.key_hash)

        if connected:
            node = None
            for n in self.connections:
                if (n.host, n.parent_port) == (host, port):
                    node = n
                    break

            self.store_key_value_pair(node.node_id, node)

    def store_key_value_pair(self, key, value):
        bucket_index = self.calculate_bucket_index(key)
        bucket = self.buckets[bucket_index]

        if not bucket.is_full():
            self.routing_table[key] = value
            bucket.add_node(self.routing_table[key])
        else:
            # Pass along to another node (x replication factor)
            target_node = self.query_routing_table(hash_key(key))

            for _ in range(self.replication_factor):
                pass

        # # Replicate the data to the next closest nodes
        # for i in range(self.replication_factor):
        #     next_node = self.get_next_node(key, node)
        #     if next_node:
        #         next_node.store(key, value)

    def forward_to_other_node(self, key, value):
        target_node = self.query_routing_table(hash_key(key))
        pickled = pickle.dumps((key, value))
        self.send_to_node(target_node, b"STORE" + pickled)

    def delete(self, key):
        """
        Delete a key-value pair from the DHT.
        """
        if key in self.routing_table:
            del self.routing_table[key]
            self.debug_print(f"Key '{key}' deleted from DHT.")
        else:
            self.debug_print(f"Key '{key}' not found in DHT.")

    def handle_statistics_request(self, callee, additional_context: dict = None):
        # memory = self.available_memory
        memory = 1e9

        stats = {"id": self.rsa_pub_key + self.port.to_bytes(4, "big"), "memory": memory}  #, "state": self.state}

        if additional_context is not None:
            for k, v in additional_context.items():
                if k not in stats.keys():
                    stats[k] = v

        stats_bytes = pickle.dumps((self.key_hash, stats))
        stats_bytes = b"RESPONSE" + stats_bytes
        self.send_to_node(callee, stats_bytes)

    def handle_peer_request(self, requesting_node):
        """
        Handle requests from other nodes to provide a list of neighboring peers.
        """
        peers = [(node.host, node.parent_port) for node in self.connections]
        message = b"RESPONSEP" + pickle.dumps(peers)
        self.send_to_node(requesting_node, message)

    # Iterate connected nodes and request their current state
    def update_worker_stats(self):
        while not self.updater_flag.is_set():

            for node in self.connections:
                # Beforehand, check the last time the worker has updated (self.prune_workers?)
                # self.request_statistics(node)
                self.request_peers()
                time.sleep(1)

            # if self.nodes:
            #     self.updater_flag.set()

            time.sleep(10)

    def request_statistics(self, worker_node):
        message = b"REQUESTS"
        self.send_to_node(worker_node, message)

    def request_peers(self):
        """
        Request neighboring nodes to send their peers.
        """
        for node in self.connections:
            message = b"REQUESTP"
            self.send_to_node(node, message)
