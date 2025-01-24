"""
Tensorlink Distributed Model Workflow Example

This script demonstrates a distributed machine learning workflow by simulating a network with a user, worker, and
validator node collaborating to coordinate the request of, and train a simple model over P2P.

Overview:
---------
1. **Node Initialization**:
   - Launches `UserNode`, `WorkerNode`, and `ValidatorNode` to simulate distributed roles.
   - `UserNode`: Manages job coordination, should be spawned in the main train or inference script.
   - `WorkerNode`: Handles data processing, can be run on local group of computers and bootstrapped to a validator
    manually, as done in this example, or through the tensorlink-miner to donate public compute.
   - `ValidatorNode`: Validates and approves tasks, coordinates resources and connections between worker and user node.
    Mainly ensures operation of the public P2P net, but is still required for local jobs.

2. **Node Connectivity**:
   - Establishes P2P connections among nodes (this process is only required for connecting local or a closed
   group of devices, network bootstrapping is handled automatically for public jobs).
   - Simulates job request through a validator and coordinates the worker with the user to create the distributed model.

3. **Model and Training**:
   - Uses `BertForSequenceClassification` for testing distributed training.
   - Demonstrates model distribution, gradient updates, and optimization across nodes.

"""

import logging
import time

import torch
from torch.nn.functional import mse_loss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)

from tensorlink import UserNode, ValidatorNode, WorkerNode

# Arg for node, when set to true network operations are on localhost (i.e. 127.0.0.1)
LOCAL = True

# Must be activated for public network use. Hopefully this is upgraded to STUN or equivalent in the near future.
UPNP = False

# Parameters for distributed model request.
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1


if __name__ == "__main__":
    # Launches a node of each type in their own process
    validator = ValidatorNode(
        upnp=UPNP, off_chain_test=LOCAL, local_test=LOCAL, print_level=logging.DEBUG
    )
    time.sleep(3)
    user = UserNode(
        upnp=UPNP, off_chain_test=LOCAL, local_test=LOCAL, print_level=logging.DEBUG
    )
    time.sleep(3)
    worker = WorkerNode(
        upnp=UPNP, off_chain_test=LOCAL, local_test=LOCAL, print_level=logging.DEBUG
    )
    time.sleep(3)

    # Get validator node information for connecting
    val_key, val_host, val_port = validator.send_request("info", None)

    # Connect worker node and user node to the validator node.
    # This would only have to be done for local jobs, and will soon be replaced by a config.json
    # For public jobs, the user will automatically be bootstrapped to the network.
    worker.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)
    user.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)

    # Create a model to distribute
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # User requests a distributed model and optimizer from a validator
    distributed_model, distributed_optimizer = user.create_distributed_model(
        model=model, training=True, optimizer_type=torch.optim.Adam
    )
    del model  # Free up some space

    # Initialize distributed optimizer
    distributed_optimizer = distributed_optimizer(lr=0.001, weight_decay=0.01)

    # Run a dummy training loop
    distributed_model.train()
    for _ in range(5):
        distributed_optimizer.zero_grad()  # Distributed optimizer calls relay to worker nodes
        x = torch.zeros((1, 1), dtype=torch.long)
        outputs = distributed_model(x)
        outputs = outputs.logits
        loss = mse_loss(outputs, outputs)
        loss.backward()
        distributed_optimizer.step()

    # Gracefully shut down nodes
    user.cleanup()
    worker.cleanup()
    validator.cleanup()
