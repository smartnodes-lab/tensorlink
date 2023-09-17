from substrateinterface import SubstrateInterface, Keypair
from src.p2p.node import Node
import subprocess
import time
import os


class SmartNode(Node):
    def __init__(self, username: str, host: str, port: int, debug: bool = False, max_connections: int = 0,
                 url: str = "wss://ws.test.azero.dev"):
        super(SmartNode, self).__init__(host, port, debug, max_connections)

        # Smart contract user parameters
        self.username = username
        self.role = 0
        self.reputations = {}
        self.keypair = self.get_public_key()
        self.chain = SubstrateInterface(url=url)

        self.init_node()

    def init_node(self):
        """
        - Grabs user data from contract (signed or encrypted?)
        - begin node and re-open any data streams with on-going connections?
        :return:
        """
        self.debug_print(f"Initializing SmartNode ({self.username})")

        # Query on chain values for user data
        result = self.chain.query(
            "user-net", "get_user_data", [self.public_key]
        )

        # Determine if user is node operator (contains the private key)
        if self.get_public_key() in result.decode():
            self.run()

    def get_public_key(self):
        path = os.getcwd()
        if not os.path.exists(os.path.join(path, "../keys/seed.pem")):
            return Keypair.create_from_uri("//Alice", self.chain.ss58_format).public_key

    def get_private_key(self):
        path = os.getcwd()
        if not os.path.exists(os.path.join(path, "../keys/seed.pem")):
            return Keypair.create_from_uri("//Alice", self.chain.ss58_format).private_key

    def create_contract(self):
        pass

    def create_p2p(self):
        super().run()

    def join_p2p(self):
        pass

    def communicate(self):
        pass

    def update_reputation(self):
        pass
