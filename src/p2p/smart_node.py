from src.auth.rsa import get_public_key_obj, authenticate_public_key
from src.p2p.node import Node

from substrateinterface import SubstrateInterface, Keypair
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

    def __init__(self, host: str, port: int, debug: bool = False, max_connections: int = 0,
                 url: str = "wss://ws.test.azero.dev", callback=None):
        super(SmartNode, self).__init__(host, port, debug, max_connections, callback)

        # Smart contract parameters
        self.chain = SubstrateInterface(url=url)
        self.keypair = self.get_substrate_keypair()
        self.contract_address = ""

        # contract_info = self.chain.query("Contracts", "ContractInfoOf", [self.contract_address])
        # if contract_info.value:
        #     self.chain.get_metadata_module()
        #     contract = ContractInstance.create_from_address(
        #         contract_address=self.contract_address,
        #         metadata_file=os.path.join(os.path.dirname(__file__), "assets", "")
        #     )

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

            # Confirm number and form conenction
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
        self.chain.query("")

    def get_substrate_keypair(self):
        # with open("../keys/README.json", "r") as f:
        #     data = json.load(f)
        #
        # encoded_key = base64.b64decode(data["encoded"]).hex()
        # return Keypair.create_from_private_key(private_key=encoded_key, ss58_format=self.chain.ss58_format)
        return Keypair.create_from_uri("//Alice")
