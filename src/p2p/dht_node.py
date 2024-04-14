from src.p2p.smart_node import SmartNode
import threading
import hashlib
import pickle
import time


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


class DHTNode(SmartNode):
    def __init__(
        self,
        host: str,
        port: int,
        public_key: str,
        debug: bool = False,
        max_connections: int = 0,
        dht_callback=None,
    ):
        super(DHTNode, self).__init__(
            host, port, public_key, debug=debug, max_connections=max_connections
        )

        # DHT Parameters
        self.replication_factor = 3
        self.bucket_size = 2
        self.routing_table = {}
        self.buckets = [Bucket(d, self.bucket_size) for d in range(256)]
        self.dht_callback = dht_callback

    def stream_data(self, data: bytes, node):
        """
        Handle incoming data streams from connected nodes and process requests.
        """
        try:
            handled = super().stream_data(data, node)

            if not handled:

                if b"STORE" == data[:5]:
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

            else:
                return True

        except Exception as e:
            self.debug_print(f"Error handling stream data: {e}")

    def run(self):
        # Thread for handling incoming connections
        listener = threading.Thread(target=self.listen, daemon=True)
        listener.start()

        print("Node stopping...")
        for node in self.connections:
            node.stop()

        time.sleep(1)

        for node in self.connections:
            node.join()

        self.sock.settimeout(None)
        self.sock.close()
        print("Node stopped")

    def calculate_bucket_index(self, key_int):
        """
        Find the index of a bucket given the key
        """
        bucket_index = key_int % len(self.buckets)

        return bucket_index

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

        # If we could not retrieve the stored value, route to nearest node
        if isinstance(closest_node[1], dict):
            # If the query matches the node id, return node info
            if closest_node[0] == key_hash:
                return closest_node[1]

            # If the query doesn't match node id, route request thru nearest node
            else:
                start_time = time.time()
                node = self.nodes[closest_node[0]]
                self.request_value(key_hash, node)

                while key_hash not in self.routing_table.keys():
                    if (
                        time.time() - start_time > 3
                    ):  # Some arbitrary timeout time for now
                        if len(ids_to_exclude) >= 1:
                            return None

                        ids_to_exclude.append(closest_node[0])
                        return self.query_routing_table(
                            key_hash,
                            ids_to_exclude,
                        )

        else:
            pass

        # In the case we have the target query value that isn't a node, return the value
        return self.routing_table[key_hash]

    def request_value(self, key: bytes, node: Connection):
        data = b"ROUTEREQ" + key
        self.send_to_node(node, data)

    def connect_dht_node(
        self, host: str, port: int, reconnect: bool = False, connected=None
    ) -> bool:
        """
        Connect to a DHT node and exchange information to identify its node ID.
        """
        if connected is None:
            connected = self.connect_with_node(host, port, reconnect, self.key_hash)

        if connected:
            node = None
            for n in self.connections:
                if (n.host, n.parent_port) == (host, port) or (n.host, n.port) == (
                    host,
                    port,
                ):
                    node = n
                    break

            if node is None:
                return False

            self.store_key_value_pair(
                node.node_id, {"host": node.host, "port": node.port}
            )
            self.nodes[node.node_id] = node
            return True

        return False

    def store_key_value_pair(self, key: bytes, value):
        key_int = int(key, 16)
        bucket_index = self.calculate_bucket_index(key_int)
        bucket = self.buckets[bucket_index]

        if not bucket.is_full():
            self.routing_table[key] = value
            bucket.add_node(self.routing_table[key])
            if hasattr(value, "role"):
                if value.role == "worker":
                    self.workers.append(key)
                elif value.role == "validator":
                    self.validators.append(key)
                elif value.role == "job":
                    self.jobs.append(key)

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

    def delete(self, key):
        """
        Delete a key-value pair from the DHT.
        """
        if key in self.routing_table:
            del self.routing_table[key]
            self.debug_print(f"Key '{key}' deleted from DHT.")
        else:
            self.debug_print(f"Key '{key}' not found in DHT.")

    def bootstrap(self):
        num_validators = self.contract.functions.getValidatorIdCounter().call() - 1
        sample_size = min(num_validators, 10)  # Adjust sample size as needed

        # Randomly select sample_size validators
        # random_sample = random.sample(range(1, num_validators + 1), sample_size)
        random_sample = [1, 2, 3]

        for validatorId in random_sample:
            # Get validator information from smart contract
            _, address, id_hash, reputation, active = (
                self.contract.functions.validators(validatorId).call()
            )

            connection_info = self.query_routing_table(id_hash.encode())
            host, port = connection_info["host"], connection_info["port"]

            # Connect to the validator's node and exchange information
            connected = self.connect_dht_node(host, port)

            # Check to see if connected, if not we can try another random node
            # if connected:
