from tensorlink.mpc.nodes import WorkerNode, ValidatorNode, UserNode
from tensorlink.crypto.rsa import *
import hashlib
from transformers import BertModel
import time


if __name__ == "__main__":

    user = UserNode(upnp=True, off_chain_test=True)
    time.sleep(0.2)
    # worker = WorkerNode(upnp=False, off_chain_test=True)
    # time.sleep(0.2)
    # validator = ValidatorNode(debug=True, upnp=False, off_chain_test=True)
    # time.sleep(0.2)
    #
    # val_key, val_host, val_port = validator.send_request("info", None)
    #
    # worker.send_request("connect_node", (val_key, val_host, val_port))
    # time.sleep(3)
    # user.send_request("connect_node", (val_key, val_host, val_port))
    # time.sleep(3)
    # user.send_request("connect_node", (b"", "142.188.24.158", 38751))
    # time.sleep(3)

    # validator.cleanup()
    # worker.cleanup()
    user.cleanup()
