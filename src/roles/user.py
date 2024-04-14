from src.ml.distributed import DistributedModel
from src.ml.model_analyzer import *
from src.p2p.connection import Connection
from src.p2p.torch_node import TorchNode

import torch.nn as nn
import threading
import pickle
import random
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
        super(User, self).__init__(
            host,
            port,
            wallet_address,
            debug=debug,
            max_connections=max_connections,
        )

        self.role = b"U"

    def stream_data(self, data: bytes, node: Connection) -> bool:
        try:
            handled = super().stream_data(data, node)

            if not handled:
                if b"ACCEPTJOB" == data[:9]:
                    recruited_workers = pickle.loads(data[9:])
                    for mod_id, worker_info in recruited_workers:
                        self.connect_dht_node(worker_info["host"], worker_info["port"])
                        self.node_requests[tuple(mod_id)].append(worker_info["id"])
                else:
                    return False
            return True

        except Exception as e:
            self.debug_print(f"worker:stream_data:{e}")
            raise e

    def request_peers(self):
        pass

    def request_job(self, model: nn.Module, max_module_size=None):
        job_id = hashlib.sha256(os.urandom(256)).hexdigest()
        model_memory = estimate_memory(model)

        if max_module_size is None:
            max_module_size = model_memory // 4

        config = self.parse_model(model, max_module_size)
        distribution = []

        for mod_id, mod_info in config.items():
            if mod_info["type"] == "offloaded":
                distribution.append((mod_info["mod_id"], mod_info["size"]))

        job_request = {
            "id": job_id,
            "author": self.key_hash,
            "capacity": 1e9,
            "dp_factor": 1,
            "distribution": distribution,  # Trainer sends list of model memory requirements for each submodule
        }

        validators = [n for n in list(self.nodes.values()) if n.role == b"V"]
        # sample_size = min(len(validators), 5)
        sample_size = 1
        random_sample = random.sample(range(len(validators)), sample_size)

        for module, mem in distribution:
            self.node_requests[tuple(module)] = []

        job_req_threads = []
        for i in random_sample:
            validator = validators[i]

            # t = threading.Thread(
            #     target=self.send_job_req, args=(validator, job_request)
            # )
            # t.start()
            # job_req_threads.append(t)
            self.send_job_req(validator, job_request)

        # for t in job_req_threads:
        #     t.join()

        dist_model_config = {}
        # Check that we have received all required workers (ie N-offloaded * DP factor)
        for mod_id, mem in job_request["distribution"]:
            # if len(self.node_requests[tuple(mod_id)]) == job_request["dp_factor"]:
            # pass
            # else:
            #     raise "Not enough recruited workers."
            # Finally, we can wrap and offload the modules
            while not self.node_requests[tuple(mod_id)]:
                pass

            module, name = access_module(model, mod_id)
            dist_model_config[name] = self.node_requests[tuple(mod_id)][-1]

        dist_model = DistributedModel(model, self, 1, 1, config=dist_model_config)
        return dist_model

    def send_job_req(self, validator, job_info):
        message = b"JOBREQ" + pickle.dumps(job_info)
        self.send_to_node(validator, message)

    def parse_model(self, model, max_module_size, config={}, ids: list = []):
        named_children = list(model.named_children())

        for i in range(len(named_children)):
            name, children = named_children[i]
            model_mem = estimate_memory(children)
            new_ids = ids + [i]
            module_type = f"{type(children)}".split(".")[-1].split(">")[0][:-1]

            if self.available_memory > model_mem:
                self.available_memory -= model_mem
                config[tuple(new_ids)] = {
                    "type": "loaded",
                    "module": module_type,
                    "mod_id": new_ids,
                    "size": model_mem,
                }

            elif model_mem < max_module_size:
                config[tuple(new_ids)] = {
                    "type": "offloaded",
                    "module": module_type,
                    "mod_id": new_ids,
                    "size": model_mem,
                }

            else:
                sub_config = self.parse_model(
                    children, max_module_size, config, new_ids
                )
                config[tuple(new_ids)] = sub_config

        return config
