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
CONTRACT_ADDRESS = "0x576511A2aC20e732122F7a98D9892214F1e0AAF1"


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
    ):
        super(SmartNode, self).__init__(
            host, port, debug, max_connections, self.stream_data
        )

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

        self.jobs = set()
        self.workers = set()
        self.validators = set()

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

            else:
                return False

            return True

        except Exception as e:
            self.debug_print(f"Error handling stream data: {e}")

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

        # Authenticate validator node's id is valid key
        if authenticate_public_key(connected_node_id) is True:
            start_time = time.time()

            if role == b"V":
                # Further confirmation of validator key via smart contract
                try:
                    verified_public_key = self.contract.functions.validatorIdByHash(
                        hashlib.sha256(connected_node_id).hexdigest()
                    ).call()

                    if verified_public_key < 1:
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
            connection.send(encrypt(message.encode(), self.port, connected_node_id))

            # Await response
            response = connection.recv(4096)
            latency = time.time() - start_time
            response, parent_port, node_id = response.split(b",")

            if response.decode() == randn:
                thread_client = self.create_connection(
                    connection,
                    client_address[0],
                    client_address[1],
                    node_id,
                    int(parent_port),
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
                    self.connections.append(thread_client)
                    self.connect_dht_node(
                        thread_client.host,
                        thread_client.parent_port,
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
        Listen for incoming connections and confirm via custom handshake
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
