from src.coordinator import WorkerCoordinator, ValidatorCoordinator, DistributedCoordinator

from transformers import BertModel
import multiprocessing
import torch
import time


if __name__ == "__main__":

    user = DistributedCoordinator()
    time.sleep(0.05)
    worker = WorkerCoordinator()
    time.sleep(0.05)
    validator = ValidatorCoordinator()

    val_key, val_host, val_port = validator.send_request("info", None)

    worker.send_request("connect_node", (val_key, val_host, val_port))
    user.send_request("connect_node", (val_key, val_host, val_port))

    model = BertModel.from_pretrained("bert-base-uncased")
    distributed_model = user.create_distributed_model(model, 1, 1.4e9)
    din = torch.zeros((1, 1), dtype=torch.long)
    output = distributed_model(din)
