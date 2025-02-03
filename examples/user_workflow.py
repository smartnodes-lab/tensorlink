"""
Tensorlink User Job Request and Distributed Training Example

This script demonstrates a user's job request on the public network, showcasing distributed training capabilities
by leveraging random public resources.

Overview:
---------
- Launches a `UserNode` that connects automatically to the public network.
- Uses `BertForSequenceClassification` as a sample model for testing.
- Demonstrates model distribution, gradient updates, and optimization through TensorLink's `DistributedModel` wrapper.
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
LOCAL = False

# Must be activated for public network use. Hopefully this is upgraded to STUN or equivalent in the near future.
UPNP = True

# Parameters for distributed model request.
BATCH_SIZE = 16
PIPELINES = 1  #
DP_FACTOR = 1


if __name__ == "__main__":
    # Launches a node of each type in their own process
    user = UserNode(
        upnp=UPNP, off_chain_test=LOCAL, local_test=LOCAL, print_level=logging.DEBUG
    )
    time.sleep(3)

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
