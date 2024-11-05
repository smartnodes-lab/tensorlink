from tensorlink.mpc.nodes import UserNode, WorkerNode, ValidatorNode
from useful_scripts import *

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import time
from transformers import BertTokenizer, BertForSequenceClassification
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
    # user.send_request("connect_node", (b"b", "142.188.24.158", 38752))
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

    validator.cleanup()

    # p1 = list(distributed_model.parameters())
    # p2 = list(distributed_model.parameters(load=False))
    # d = distributed_model.state_dict()
    distributed_optimizer = distributed_optimizer(lr=0.001, weight_decay=0.01)

    for _ in range(10):
        distributed_optimizer.zero_grad()
        x = torch.zeros((10, 10))
        x = distributed_model(x)
        loss = mse_loss(x, x)
        loss.backward()
        distributed_optimizer.step()

    # train(distributed_model, distributed_optimizer, tokenizer, device, batch_size=BATCH_SIZE)
    user.cleanup()
    worker.cleanup()
