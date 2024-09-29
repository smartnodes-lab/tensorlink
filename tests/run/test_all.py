import hashlib

from src.mpc.coordinator import WorkerCoordinator, ValidatorCoordinator, DistributedCoordinator
from src.crypto.rsa import *

from transformers import BertModel
import torch
import time


if __name__ == "__main__":

    user = DistributedCoordinator()
    time.sleep(0.2)
    # worker = WorkerCoordinator()
    # time.sleep(0.2)
    # validator = ValidatorCoordinator()

    # Additional tweaks for key access, override for running 2 validators on same device
    # validator.node_process.rsa_pub_key = get_public_key_bytes(
    #     get_rsa_pub_key(b"V2")
    # )
    # validator.node_process.rsa_key_hash = hashlib.sha256(validator.node_process.rsa_pub_key)
    time.sleep(0.2)

    # val_key, val_host, val_port = validator.send_request("info", None)

    # worker.send_request("connect_node", (val_key, val_host, val_port))
    user.send_request("connect_node", (val_key, val_host, val_port))

    # while True:
    #     pass

    model = BertModel.from_pretrained("bert-base-uncased")
    distributed_model = user.create_distributed_model(model, 1, 1)
    for _ in range(10):
        din = torch.zeros((1, 32), dtype=torch.long)
        output = distributed_model(din)
    # loss = output.last_hidden_state
    # distributed_model.backward(loss)
