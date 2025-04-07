"""To be used as a sandbox for interacting with the main node framework"""

from tensorlink import ValidatorNode, WorkerNode
import logging
import time


UPNP = True
OFFCHAIN = False
LOCAL = False


if __name__ == "__main__":
    node = WorkerNode(
        upnp=UPNP,
        off_chain_test=OFFCHAIN,
        local_test=LOCAL,
        print_level=logging.DEBUG,
        utilization=False,
    )

    while True:
        time.sleep(5)
