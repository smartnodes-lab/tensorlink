from src.ml.distributed import DistributedModel
from src.ml.model_analyzer import handle_output
from src.ml.worker import Worker

from transformers import BertModel
import torch.nn as nn
import torch
import time
import os


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5026

    master = Worker(host=ip, port=port, debug=True)
    worker1 = Worker(host=ip, port=port + 1, debug=True)

    master.master = True
    master.training = True
    worker1.training = True

    master.start()
    worker1.start()
    master.connect_with_node(ip, port + 1)
    worker1.connect_with_node(ip, port)

    model = BertModel.from_pretrained("bert-base-uncased")
    dummy_input = torch.zeros((1, 1), dtype=torch.long)

    nodes = [
        {"id": 1, "memory": 1.4e9, "connection": master.outbound[0], "latency_matrix": []}
    ]

    distributed = DistributedModel(master, model, nodes)
    distributed.create_distributed_model()

    distributed.master_node.intermediates.append([dummy_input])  # To be moved to model source code
    out = distributed.model.forward(dummy_input)

    loss = handle_output(out).sum()

    distributed.backward(loss)

    master.stop()
    worker1.stop()
