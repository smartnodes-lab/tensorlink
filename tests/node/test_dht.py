import hashlib
import time
from multiprocessing import Queue
from src.nodes.validator import Validator
from src.nodes.user import User
from src.crypto.rsa import *
# from src.mpc.coordinator import ValidatorCoordinator


node1 = Validator(Queue(), Queue(), upnp=False, off_chain_test=True, debug=True)
node2 = Validator(Queue(), Queue(), upnp=False, off_chain_test=True, debug=True)
pk = get_rsa_pub_key(b"V2")
node2.rsa_pub_key = get_public_key_bytes(pk)
node2.rsa_key_hash = hashlib.sha256(node2.rsa_pub_key)
node2.role = b"V2"
user = User(Queue(), Queue(), upnp=False, off_chain_test=True, debug=True)
node1.start()
node2.start()
user.start()

node2.connect_node(node1.rsa_key_hash, node1.host, node1.port)
user.connect_node(node1.rsa_key_hash, node1.host, node1.port)
# node2.ping_node(node2.nodes[node1.rsa_key_hash])
user.request_job(3, 1, [1e9, 1e9])

time.sleep(1)

node1.stop()
node2.stop()
