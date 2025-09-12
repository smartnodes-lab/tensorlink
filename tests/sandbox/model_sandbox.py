"""A sandbox for experimenting locally with models"""

from tensorlink import WorkerNode, UserNode, ValidatorNode, DistributedModel
import logging
import torch
import time

OFFCHAIN = False
LOCAL = True
UPNP = False


if __name__ == "__main__":
    # Launches a node of each type in their own process
    validator = ValidatorNode(
        upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG
    )
    # Small sleep for preventing two nodes from starting on the same port and conflicting
    time.sleep(0.1)
    user = UserNode(
        upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG
    )
    time.sleep(0.1)
    worker = WorkerNode(
        upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG
    )
    time.sleep(0.1)

    # Get validator node information for connecting
    val_key, val_host, val_port = validator.send_request("info", None)
    worker.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)
    user.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(3)

    # Initialize the distributed model job
    distributed_model = DistributedModel("bert-base-uncased", node=user, training=False)

    outputs = distributed_model.forward(torch.zeros((1, 1), dtype=torch.long))

    while True:
        time.sleep(3)

    user.cleanup()
    worker.cleanup()
    validator.cleanup()
