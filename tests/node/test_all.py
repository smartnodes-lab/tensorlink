from tensorlink.mpc.nodes import WorkerNode, ValidatorNode, UserNode
from tensorlink.crypto.rsa import *
import hashlib
from transformers import BertModel
import time


if __name__ == "__main__":
    # Run each node type and connect to each other (good for a test node sandbox)
    user = UserNode(upnp=False, off_chain_test=True, print_level=10)
    time.sleep(1)
    worker = WorkerNode(upnp=False, off_chain_test=True, print_level=10)
    time.sleep(1)
    validator = ValidatorNode(upnp=False, off_chain_test=True, print_level=10)
    time.sleep(1)

    val_key, val_host, val_port = validator.send_request("info", None)

    worker.send_request("connect_node", (val_key, val_host, val_port))
    time.sleep(3)
    user.send_request("connect_node", (val_key, val_host, val_port))
    time.sleep(3)

    # An optional loop for debugging nodes
    # try:
    #     while True:
    #         time.sleep(3)
    # except KeyboardInterrupt:
    #     pass

    user.cleanup()
    validator.cleanup()
    worker.cleanup()
