"""A sandbox for experimenting with interactions between nodes"""

from tensorlink import WorkerNode, UserNode, ValidatorNode
import logging
import time


OFFCHAIN = False
LOCAL = True
UPNP = False


if __name__ == "__main__":
    # Launches a node of each type in their own process
    validator = ValidatorNode(
        upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG
    )
    # Temporary sleep for preventing two nodes from starting on the same port and conflicting
    time.sleep(1)
    # user = UserNode(
    #     upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG
    # )
    # time.sleep(1)
    worker = WorkerNode(
        upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG
    )
    time.sleep(1)

    # Get validator node information for connecting
    val_key, val_host, val_port = validator.send_request("info", None)
    worker.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)
    # user.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    # time.sleep(1)

    while True:
        time.sleep(3)
