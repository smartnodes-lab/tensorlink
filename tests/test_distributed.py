import torch

from src.ml.distributed import DistributedModel

from transformers import BertModel
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import torch.nn as nn
import time
import os


os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "5028"


if __name__ == "__main__":

    model = BertModel.from_pretrained("bert-base-uncased")
    # model = nn.Sequential(nn.Linear(10, 6000000), nn.Linear(6000000, 2))

    model = DistributedModel(model)

    # model = nn.ModuleList([
    #     nn.Linear(1, 10),
    #     nn.Linear(10, 10),
    #     nn.Linear(10, 10)
    # ])
    #
    # submodules = list(model.children())
    # world_size = 3
    # tik = time.time()
    #
    # mp.spawn(run_worker, args=(world_size, submodules), nprocs=world_size)
    #
    # tok = time.time()
    # print(f"Execution time: {round(tok - tik, 1)}s")

