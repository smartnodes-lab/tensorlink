"""
test_local_dist_train.py

This script tests distributed machine learning using Tensorlink's P2P network using local nodes on different
processes. It simulates a local environment with a user, worker, and validator node collaborating to train a
simple model.

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
from tensorlink import UserNode, WorkerNode, ValidatorNode
import pytest
import logging
import time


# Variables for nodes and distributed models
LOCAL = True
UPNP = False
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1


@pytest.fixture(scope="module")
def nodes():
    """Initialize and return validator, user, and worker nodes."""
    validator = ValidatorNode(upnp=UPNP, off_chain_test=LOCAL, local_test=LOCAL, print_level=logging.DEBUG)
    time.sleep(3)
    user = UserNode(upnp=UPNP, off_chain_test=LOCAL, local_test=LOCAL, print_level=logging.DEBUG)
    time.sleep(3)
    worker = WorkerNode(upnp=UPNP, off_chain_test=LOCAL, local_test=LOCAL, print_level=logging.DEBUG)
    time.sleep(3)

    yield validator, user, worker

    # Cleanup nodes after tests
    user.cleanup()
    worker.cleanup()
    validator.cleanup()


def test_node_initialization(nodes):
    """Test that nodes are initialized correctly."""
    validator, user, worker = nodes
    assert validator is not None, "Validator node failed to initialize."
    assert user is not None, "User node failed to initialize."
    assert worker is not None, "Worker node failed to initialize."


def test_node_connectivity(nodes):
    """Test that nodes connect successfully."""
    validator, user, worker = nodes
    val_key, val_host, val_port = validator.send_request("info", None)

    # Worker connects to validator
    worker.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)

    # User connects to validator
    user.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)

    assert val_key is not None, "Validator key is None."
    assert val_host is not None, "Validator host is None."
    assert val_port is not None, "Validator port is None."


def test_distributed_training(nodes):
    """Test distributed training with a simple model."""
    _, user, _ = nodes

    # Create model and tokenizer
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.nn.functional import mse_loss
    import torch

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    distributed_model, distributed_optimizer = user.create_distributed_model(
        model=model,
        training=True,
        optimizer_type=torch.optim.Adam
    )
    del model  # Free up memory
    distributed_optimizer = distributed_optimizer(lr=0.001, weight_decay=0.01)

    # Training loop
    distributed_model.train()
    for _ in range(5):
        distributed_optimizer.zero_grad()
        x = torch.zeros((1, 1), dtype=torch.long)
        outputs = distributed_model(x)
        loss = mse_loss(outputs.logits, outputs.logits)
        loss.backward()
        distributed_optimizer.step()

    assert distributed_model is not None, "Distributed model is None."
    assert distributed_optimizer is not None, "Distributed optimizer is None."
