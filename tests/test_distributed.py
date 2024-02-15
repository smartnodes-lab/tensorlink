from src.ml.distributed import DistributedModel, print_distribute_model
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

    # worker1 = Worker(host=ip, port=port, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    #                  debug=True)
    # worker2 = Worker(host=ip, port=port + 1, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    #                  debug=True)
    # worker3 = Worker(host=ip, port=port + 2, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    #                  debug=True)
    #
    # worker1.master = True
    # worker1.training = True
    # worker2.training = True
    # worker3.training = True
    #
    # worker1.start()
    # worker2.start()
    # worker3.start()
    #
    # # worker1.connect_with_node(ip, port + 1)
    # worker2.connect_with_node(ip, port)
    # worker3.connect_with_node(ip, port)
    # worker3.connect_with_node(ip, port + 1)
    #
    # worker1.update_statistics()
    # worker2.update_statistics()
    # worker3.update_statistics()
    #
    # dummy_input = torch.zeros((1, 1), dtype=torch.long)
    model = BertModel.from_pretrained("bert-base-uncased")
    #
    # distributed = DistributedModel(worker1, model, worker1.peer_stats)
    # print("DONE")
    # time.sleep(20)
    # distributed(dummy_input)
    #
    # worker1.stop()
    # worker2.stop()
    # worker3.stop()

    # distributed = DistributedModel(worker1, model)
    # distributed.create_distributed_model(model)
    print_distribute_model(model)
