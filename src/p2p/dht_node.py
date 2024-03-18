import threading

from src.p2p.smart_node import SmartNode
import hashlib
import pickle
import queue
import time
import os


class DHTNode(SmartNode):
    def __init__(self, host: str, port: int, wallet_address: str, debug: bool = False, max_connections: int = 0):
        super(DHTNode, self).__init__(
            host, port, wallet_address, debug=debug, max_connections=max_connections, callback=self.stream_data
        )

        self.routing_table = {}  # Key-value store
        self.replication_factor = 3  # Number of replicas for each key
        self.key_hash = hashlib.sha256(self.rsa_pub_key).hexdigest()

        self.updater_flag = threading.Event()

    def query_routing_table(self, key):
        """
        Get the node responsible for a given key.
        """
        key_hash = self.hash_key(key)
        closest_node = None
        closest_distance = float('inf')
        for node in self.connections:
            distance = self.calculate_xor_distance(key_hash, node.node_id)
            if distance < closest_distance:
                closest_node = node
                closest_distance = distance

        return closest_node

    def run(self):
        """

        """
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
                self.store(key, value)

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
                    self.connect_with_node(host, port)

            # Add more data types and their handling logic as needed

        except Exception as e:
            self.debug_print(f"Error handling stream data: {e}")

    def hash_key(self, key):
        """
        Hashes the key to determine its position in the keyspace.
        """
        key_bytes = key.encode('utf-8')
        return int.from_bytes(hashlib.sha1(key_bytes).digest(), byteorder='big')

    def calculate_xor_distance(self, key_hash, node_id):
        """
        Calculate the XOR distance between a key and a node ID.
        """
        return key_hash ^ node_id

    def store(self, key, value):
        """
        Store a key-value pair in the DHT.
        """
        node = self.query_routing_table(key)
        self.routing_table[key] = value
        # Replicate the data to the next closest nodes
        for i in range(1, self.replication_factor):
            next_node = self.get_next_node(key, node)
            if next_node:
                next_node.store(key, value)

    def get_next_node(self, key, current_node, key_hash=None):
        """
        Get node closest to us via xor distance. TODO: account for latency as well
        """
        if key_hash is None:
            key_hash = self.hash_key(key)

        closest_distance = float('inf')
        next_node = None
        for node in self.connections:
            if node != current_node:
                distance = self.calculate_xor_distance(key_hash, node.node_id)
                if distance < closest_distance:
                    closest_distance = distance
                    next_node = node

        return next_node

    def retrieve(self, key):
        """
        Retrieve the value associated with a key from the DHT. If we don't have, get the closest node
        """
        if key in self.routing_table:
            return self.routing_table[key]
        else:
            node = self.query_routing_table(key)
            return node.retrieve(key)

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
