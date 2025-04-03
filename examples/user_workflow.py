"""
Tensorlink User Job Request and Distributed Training Example

This script demonstrates a user's job request on the public network, showcasing distributed training capabilities
by leveraging random public resources.


Model distribution, gradient updates, and optimization through TensorLink's `DistributedModel` wrapper.
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

from tensorlink import DistributedModel


# Parameters for distributed model request.
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1


if __name__ == "__main__":
    # Get distributed model directly from HuggingFace without loading
    distributed_model = DistributedModel(
        'bert-base-uncased',  # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        training=False,
        n_pipelines=PIPELINES,
    )

    # Alternatively, you could load a model to distribute (for hybrid jobs and custom models)
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # distributed_model = DistributedModel(
    #     model=model, optimizer_type=torch.optim.Adam, training=False
    # )
    # del model

    # Initialize distributed optimizer
    # distributed_optimizer = distributed_model.create_optimizer(
    #     lr=0.001, weight_decay=0.01
    # )

    # Run a dummy training loop
    # distributed_model.train()
    for _ in range(5):
        # distributed_optimizer.zero_grad()  # Distributed optimizer calls relay to worker nodes
        x = torch.zeros((1, 1), dtype=torch.long)
        outputs = distributed_model(x)
        outputs = outputs.logits
        loss = mse_loss(outputs, outputs)
        # loss.backward()
        # distributed_optimizer.step()
