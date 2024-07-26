from src.mpc.coordinator import DistributedCoordinator, WorkerCoordinator, ValidatorCoordinator
from useful_scripts import *

import torch
import time
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    set_seed,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import logging

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


if __name__ == "__main__":
    # Launch Nodes
    user = DistributedCoordinator(debug=True)
    time.sleep(0.2)
    worker = WorkerCoordinator(debug=True)
    time.sleep(0.2)
    validator = ValidatorCoordinator(debug=True)

    # Bootstrap nodes
    val_key, val_host, val_port = validator.send_request("info", None)
    worker.send_request("connect_node", (val_key, val_host, val_port))
    user.send_request("connect_node", (val_key, val_host, val_port))
    user.send_request("connect_node", (b'58ef79797cd451e19df4a73fbd9871797f9c6a2995783c7f6fd2406978a2ba2e', "192.168.2.64", 38752))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    ).to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    distributed_model = user.create_distributed_model(model, 1, 1.4e9)
    train(distributed_model, tokenizer, device, logger)
