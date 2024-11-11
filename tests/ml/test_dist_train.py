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
from transformers import BertTokenizer, BertForSequenceClassification
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


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Launch Nodes
    user = UserNode(upnp=False, debug=True, off_chain_test=True, local_test=False, print_level=logging.DEBUG)
    time.sleep(0.5)
    worker = WorkerNode(upnp=False, debug=True, off_chain_test=True, local_test=False, print_level=logging.DEBUG)
    time.sleep(0.5)
    validator = ValidatorNode(upnp=False, debug=True, off_chain_test=True, local_test=False, print_level=logging.DEBUG)
    time.sleep(0.5)

    # Bootstrap roles
    val_key, val_host, val_port = validator.send_request("info", None)

    worker.send_request("connect_node", (val_key, val_host, val_port))
    time.sleep(1)

    user.send_request("connect_node", (val_key, val_host, val_port))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it",
    #                                           token="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")
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
    #                                           token="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")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model = Dummy()

    distributed_model, distributed_optimizer = user.create_distributed_model(
        model=model,
        # max_input_size=(2, 2),
        # max_batch_size=64,
        n_pipelines=PIPELINES,
        optimizer_type=torch.optim.Adam,
        dp_factor=1
    )
    del model

    # p1 = list(distributed_model.parameters())
    # p2 = list(distributed_model.parameters(load=False))
    d = distributed_model.state_dict()

    # distributed_optimizer = distributed_optimizer(lr=0.001, weight_decay=0.01)
    # distributed_model.train()
    #
    # for _ in range(1):
    #     distributed_optimizer.zero_grad()
    #     x = torch.zeros((1, 10), dtype=torch.float)
    #     outputs = distributed_model(x)
    #     # outputs = outputs.logits
    #     loss = mse_loss(outputs, outputs)
    #     loss.backward()
    #     distributed_optimizer.step()

    # time.sleep(300)
    user.cleanup()
    validator.cleanup()
    worker.cleanup()
