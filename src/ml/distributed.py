import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import time
import os


def run_master(splits):
    pass


def run_worker(rank, world_size):
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

        # run_master()

    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    rpc.shutdown()
