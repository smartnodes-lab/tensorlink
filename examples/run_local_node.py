"""
This script demonstrates launching of, and connecting to a node
"""

from tensorlink import UserNode, ValidatorNode, WorkerNode
import logging
import time


LOCAL = True  # Network operations are on localhost when true (i.e. 127.0.0.1)
UPNP = (
    True if not LOCAL else False
)  # Must be activated for public network use. Upgrade to STUN or other in the future.
OFFCHAIN = LOCAL  # Can be used to deactivate on-chain features (for private jobs)

# Parameters for distributed model request.
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1
TRAINING = False  # Set true to request train job and get a distributed optimizer

# Chatbot parameters
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
MAX_HISTORY_TURNS = 6
MAX_TOKENS = 2048
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7


if __name__ == "__main__":
    # Launches a node of each type in their own process (Not necessary if just accessing a DistributedModel
    # as a user, it will do this in the background...
    validator = ValidatorNode(
        upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG
    )
    worker = WorkerNode(
        upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG
    )

    worker.connect_node()

    while True:
        try:
            time.sleep(1)

        except KeyboardInterrupt or Exception:
            break

    # Gracefully shut down nodes
    worker.cleanup()
    validator.cleanup()
