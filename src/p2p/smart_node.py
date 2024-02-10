from src.auth.rsa import get_public_key_obj, authenticate_public_key
from src.auth.substrate_keys import load_substrate_keypair
from src.p2p.node import Node

from substrateinterface import SubstrateInterface
from substrateinterface.contracts import ContractCode, ContractInstance, ContractMetadata
import random
import socket
import time
import os


class SmartNode(Node):
    """
    TODO:
    - confirm workers public key with smart contract ID
    """

    def __init__(self, host: str, port: int, public_key: str, url: str = "wss://ws.test.azero.dev",
                 contract: str = "5EYpWTZahNC6ko7nmEAVZrnWsW39tRRDX3UhPKDcsFQeQQMh", debug: bool = False,
                 max_connections: int = 0, callback=None,):
        super(SmartNode, self).__init__(host, port, debug, max_connections, callback)

        # Smart contract parameters
        self.chain = SubstrateInterface(url=url)
        self.keypair = load_substrate_keypair(public_key, "      ")
        self.contract_address = contract
        self.contract = None

        # Grab the SmartNode contract
        contract_info = self.chain.query("Contracts", "ContractInfoOf", [self.contract_address])
        if contract_info.value:
            self.contract = contract_info

    def handshake(self, connection, client_address):
        """
        Validates incoming connection's keys with a random number swap
        """
        connected_node_id = connection.recv(4096)

        # Generate random number to confirm with incoming node
        randn = str(random.random())
        message = randn

        # Authenticate incoming node's id is valid key
        if authenticate_public_key(connected_node_id) is True:
            id_bytes = connected_node_id

            # Encrypt random number with node's key
            connection.send(
                self.encrypt(message.encode(), id_bytes)
            )

            # Await response
            response = connection.recv(4096)

            # Confirm number and form connection
            if response.decode() == randn:
                thread_client = self.create_connection(connection, connected_node_id, client_address[0],
                                                       client_address[1])
                thread_client.start()

                self.inbound.append(thread_client)
                self.outbound.append(thread_client)

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
                if self.max_connections == 0 or len(self.inbound) < self.max_connections:
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

    def get_job(self):
        # Confirm job details with smart contract, receive initial details from a node?
        # self.chain.query("")
        pass

    # def get_user_info(self, user_address):
    #
    #     try:
    #         user_info = self.chain.compose_call(
    #             call_module="",
    #             call_function="",
    #             call_params={}
    #         )
    #     pass
