from src.ml.distributed import run_worker, distribute_model

from transformers import BertModel
import torch.multiprocessing as mp
import time
import os


os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "5028"


if __name__ == "__main__":
    model = BertModel.from_pretrained("bert-base-uncased")
    # distribute_model(model)
    submodules = list(model.children())
    world_size = 3
    tik = time.time()
    mp.spawn(run_worker, args=(world_size, submodules), nprocs=world_size)
    tok = time.time()
    print(f"Execution time: {round(tok - tik, 1)}s")
