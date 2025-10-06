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

import logging
import time

import pytest

from tensorlink import UserNode, ValidatorNode, WorkerNode

# Variables for nodes and distributed models
LOCAL = True
UPNP = False
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1


@pytest.fixture(scope="module")
def nodes():
    """Initialize and return validator, user, and worker nodes."""
    # Cleanup nodes after tests

    validator = ValidatorNode(
        upnp=UPNP,
        off_chain_test=LOCAL,
        local_test=LOCAL,
        print_level=logging.DEBUG,
        endpoint=False,
    )  # Must turn off endpoint for pytest
    time.sleep(1)
    user = UserNode(
        upnp=UPNP, off_chain_test=LOCAL, local_test=LOCAL, print_level=logging.DEBUG
    )
    time.sleep(1)
    worker = WorkerNode(
        upnp=UPNP, off_chain_test=LOCAL, local_test=LOCAL, print_level=logging.DEBUG
    )
    time.sleep(1)

    try:
        yield validator, user, worker

    except Exception as e:
        print(f"Error during node cleanup: {e}")

    finally:
        user.cleanup()
        worker.cleanup()
        validator.cleanup()

        time.sleep(3)


def test_node_initialization(nodes):
    """Test that nodes are initialized correctly and that nodes connect successfully."""
    validator, user, worker = nodes
    assert validator is not None, "Validator node failed to initialize."
    assert user is not None, "User node failed to initialize."
    assert worker is not None, "Worker node failed to initialize."

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


# def test_distributed_setup(nodes):
#     """Test distributed training with a simple model."""
#     validator, user, worker = nodes
#
#     # Create model and tokenizer
#     import torch
#     import torch.nn as nn
#
#     model = nn.ModuleList([nn.Linear(10, 10)])
#
#     distributed_model, distributed_optimizer = user.create_distributed_model(
#         model=model, training=True, optimizer_type=torch.optim.Adam
#     )
#     del model
#     distributed_optimizer = distributed_optimizer(lr=0.001, weight_decay=0.01)
#
#     distributed_model.train()
#     distributed_optimizer.zero_grad()
#
#     assert distributed_model is not None, "Distributed model is None."
#     assert distributed_optimizer is not None, "Distributed optimizer is None."
