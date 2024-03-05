from src.cryptography.rsa import *
from src.cryptography.substrate import load_substrate_keypair
from src.p2p.node import Node

from substrateinterface import SubstrateInterface
from substrateinterface.contracts import ContractCode, ContractInstance, ContractMetadata
from substrateinterface.exceptions import SubstrateRequestException

import random
import socket
import time
import os


METADATA = "./src/assets/smartnodes.json"
CONTRACT = "5D37KYdd3Ptd8CfjKxtN3rGqU6oZQQeiGfTLfF2VGrTQJfyN"


class SmartNode(Node):
    """
    TODO:
    - confirm workers public key with smart contract ID
    """
    def __init__(self, host: str, port: int, public_key: str, url: str = "wss://ws.test.azero.dev",
                 contract: str = CONTRACT, debug: bool = False,
                 max_connections: int = 0, callback=None):
        super(SmartNode, self).__init__(host, port, debug, max_connections, callback)

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

    def handshake(self, connection, client_address):
        """
        Validates incoming connection's keys via a random number swap, along with SC
            verification of connecting user
        """
        connected_node_id = connection.recv(4096)
        print(connected_node_id)

        # Generate random number to confirm with incoming node
        randn = str(random.random())
        message = randn

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

            if response.decode() == randn:
                thread_client = self.create_connection(connection, connected_node_id,
                                                       client_address[0], client_address[1])
                thread_client.start()

                thread_client.latency = latency
                self.connections.append(thread_client)

            else:
                self.debug_print("node: connection refused, invalid ID proof!")
                connection.close()
            #
            #     else:
            #         self.debug_print("User not listed on contract.")
            #
            # except SubstrateRequestException as e:
            #     self.debug_print(f"Failed to verify user public key: {e}")
            #
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
