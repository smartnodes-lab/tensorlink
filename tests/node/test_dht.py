import hashlib
import time
from multiprocessing import Queue
from tensorlink.roles.validator import Validator
from tensorlink.roles.user import User
from tensorlink.roles.worker import Worker
from tensorlink.crypto.rsa import *
# from tensorlink.mpc.coordinator import ValidatorCoordinator


node1 = Validator(Queue(), Queue(), upnp=False, off_chain_test=True, local_test=True, debug=True)
user = User(Queue(), Queue(), upnp=False, off_chain_test=True, debug=True, local_test=True)
worker = Worker(Queue(), Queue(), upnp=False, off_chain_test=True, debug=True, local_test=True)
# pk = get_rsa_pub_key(b"V2")
# node2 = Validator(Queue(), Queue(), upnp=False, off_chain_test=True, debug=True)
# node2.rsa_pub_key = get_public_key_bytes(pk)
# node2.rsa_key_hash = hashlib.sha256(node2.rsa_pub_key)
# node2.role = b"V2"
node1.start()
user.start()
worker.start()
# node2.start()

# node2.connect_node(node1.rsa_key_hash, node1.host, node1.port)
user.connect_node(node1.rsa_key_hash, node1.host, node1.port)
time.sleep(1)
worker.connect_node(node1.rsa_key_hash, node1.host, node1.port)
time.sleep(1)
# node2.connect_node(node1.rsa_key_hash, node1.host, node1.port)
# time.sleep(1)

print(
    f"Validator Address: {node1.host}:{node1.port} --- {[(node.host, node.port) for node in node1.nodes.values()]}\n",
    # f"Validator2 Address: {node2.host}:{node2.port} --- {[(node.host, node.port) for node in node2.nodes.values()]}\n"
    f"Worker Address: {worker.host}:{worker.port} --- {[(node.host, node.port) for node in worker.nodes.values()]}\n",
    f"User Address: {user.host}:{user.port} --- {[(node.host, node.port) for node in user.nodes.values()]}\n",
)
# node2.ping_node(node2.roles[node1.rsa_key_hash])
# user.request_job(3, 1, [1e9, 1e9])

while True:
    try:
        time.sleep(1)

    except KeyboardInterrupt:
        break

node1.stop()
user.stop()
worker.stop()
# node2.stop()
