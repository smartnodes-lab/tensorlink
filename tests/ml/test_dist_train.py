"""
TensorLink Job Distribution Test Script

This script demonstrates how a machine learning job is distributed across the tensorlink network using off-chain and local connections on the user's PC. It simulates a distributed setup where user, worker, and validator nodes collaborate to perform training on a simple model. The script utilizes TensorLink's networking features to establish connections among the nodes and set up a basic distributed machine learning workflow.

Description:
------------
1. Node Initialization:
    - Launches `UserNode`, `WorkerNode`, and `ValidatorNode` instances.
    - Each node simulates different roles in a distributed system:
      - `UserNode`: Initiates and coordinates the job.
      - `WorkerNode`: Processes data as part of the distributed model.
      - `ValidatorNode`: Verifies and approves job-related tasks.

2. Node Connectivity:
    - Establishes peer-to-peer connections among the nodes.
    - Uses off-chain testing to simulate realistic job distribution without requiring a blockchain connection.
    - Connects `WorkerNode` and `UserNode` to the `ValidatorNode`.

3. Model and Optimizer Setup:
    - Defines a basic neural network model (`Dummy`) to test the distributed training setup.
    - Uses `create_distributed_model()` from the `UserNode` to distribute the model across the nodes.
    - Sets up an optimizer for training, demonstrating distributed gradient updates and optimization.

4. Distributed Training:
    - Runs a training loop on the distributed model.
    - Demonstrates how the `UserNode`, `WorkerNode`, and `ValidatorNode` interact to handle backpropagation and model updates.
    - Logs training progress, including model connection details, loss calculations, and optimizer steps.
"""
from tensorlink import UserNode, WorkerNode, ValidatorNode
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import mse_loss
import torch.nn as nn
import torch
import time
import logging
import json

# Set up logging
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('training.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


BATCH_SIZE = 64
PIPELINES = 1
DP_FACTOR = 1


if __name__ == "__main__":
    # Launch Nodes
    validator = ValidatorNode(upnp=True, off_chain_test=False, local_test=False, print_level=logging.DEBUG)
    time.sleep(1)
    user = UserNode(upnp=True, off_chain_test=False, local_test=False, print_level=logging.DEBUG)
    time.sleep(1)
    worker = WorkerNode(upnp=True, off_chain_test=False, local_test=False, print_level=logging.DEBUG)
    time.sleep(1)

    # Bootstrap roles
    # val_key, val_host, val_port = validator.send_request("info", None)
    #
    # worker.send_request("connect_node", (val_key, val_host, val_port))
    # time.sleep(1)
    # user.send_request("connect_node", (val_key, val_host, val_port))
    # time.sleep(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    distributed_model, distributed_optimizer = user.create_distributed_model(
        model=model,
        training=True,
        optimizer_type=torch.optim.Adam
    )
    del model

    # p1 = list(distributed_model.parameters())
    # p2 = list(distributed_model.parameters(load=False))
    # d = distributed_model.state_dict()

    distributed_optimizer = distributed_optimizer(lr=0.001, weight_decay=0.01)
    distributed_model.train()

    for _ in range(2):
        distributed_optimizer.zero_grad()
        x = torch.zeros((1, 1), dtype=torch.long)
        outputs = distributed_model(x)
        outputs = outputs.logits
        loss = mse_loss(outputs, outputs)
        loss.backward()
        distributed_optimizer.step()

    user.cleanup()
    worker.cleanup()
    validator.cleanup()
