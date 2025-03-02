"""To be used as a sandbox for interacting with the main node framework"""

from tensorlink import ValidatorNode
import time


UPNP = True
OFFCHAIN = False
LOCAL = False


if __name__ == "__main__":
    node = ValidatorNode(
        upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=10
    )

    while True:
        time.sleep(5)
