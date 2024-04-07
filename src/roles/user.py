from src.ml.distributed import DistributedModel
from src.p2p.torch_node import TorchNode

import torch.nn as nn
import hashlib
import os


class User(TorchNode):
    def __init__(
        self,
        host: str,
        port: int,
        wallet_address: str,
        debug: bool = False,
        max_connections: int = 0,
    ):
        super(TorchNode, self).__init__(
            host,
            port,
            wallet_address,
            debug=debug,
            max_connections=max_connections,
        )

    def request_peers(self):
        pass

    def request_job(self, model: nn.Module):
        job_id = hashlib.sha256(os.urandom(256)).hexdigest()

        job_request = {
            "job": job_id,
            "author": self.key_hash,
            "capacity": 0,
            "dp_factor": 3,
            "distribution": [
                1e9,
                1e9,
                1e9,
                1e9,
            ],  # Trainer sends list of model memory requirements for each
            # submodule
        }
