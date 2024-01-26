from src.ml.distributed import run_worker

import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os


os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "5028"


if __name__ == "__main__":
    world_size = 2
    tik = time.time()
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size)
    tok = time.time()
    print(f"Execution time: {round(tok - tik, 1)}s")
