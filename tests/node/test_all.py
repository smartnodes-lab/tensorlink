from tensorlink.mpc.nodes import WorkerNode, ValidatorNode, UserNode
from tensorlink.crypto.rsa import *
import hashlib
from transformers import BertModel
import time


if __name__ == "__main__":

    user = UserNode(upnp=False, off_chain_test=True, print_level=10)
    time.sleep(1)
    # worker = WorkerNode(upnp=False, off_chain_test=True, print_level=10)
    # time.sleep(1)
    validator = ValidatorNode(upnp=False, off_chain_test=True, print_level=10)
    time.sleep(1)

    val_key, val_host, val_port = validator.send_request("info", None)

    # worker.send_request("connect_node", (val_key, val_host, val_port))
    # time.sleep(3)
    user.send_request("connect_node", (val_key, val_host, val_port))
    time.sleep(3)
    # user.send_request("connect_node", (b"", "142.188.24.158", 38751))
    # time.sleep(3)

    user.cleanup()
    # validator.cleanup()
    # worker.cleanup()
    while True:
        time.sleep(1)
