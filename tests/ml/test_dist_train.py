from src.mpc.nodes import DistributedCoordinator, WorkerCoordinator, ValidatorCoordinator
from useful_scripts import *

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    user = DistributedCoordinator(upnp=False, debug=True, off_chain_test=True)
    time.sleep(0.5)
    # worker = WorkerCoordinator(upnp=False, debug=True, off_chain_test=True)
    # time.sleep(0.5)
    # validator = ValidatorCoordinator(upnp=False, debug=True, off_chain_test=True)
    # time.sleep(0.5)

    # Bootstrap roles
    # val_key, val_host, val_port = validator.send_request("info", None)

    # while True:
    #     pass

    # worker.send_request("connect_node", (val_key, val_host, val_port))
    user.send_request("connect_node", (b"b", "142.188.24.158", 38757))
    # user.send_request("connect_node", (val_key, val_host, val_port))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it",
    #                                           token="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")
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
    #                                           token="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")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
    distributed_model = user.create_distributed_model(model, PIPELINES, DP_FACTOR)
    del model
    train(distributed_model, tokenizer, device, logger, BATCH_SIZE)
