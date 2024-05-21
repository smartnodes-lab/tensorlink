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
        private_key: str,
        debug: bool = False,
        max_connections: int = 0,
        upnp=True,
    ):
        super(User, self).__init__(
            host,
            port,
            private_key,
            debug=debug,
            max_connections=max_connections,
            upnp=upnp,
        )

        self.role = b"U"
        self.job = None

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

    def request_job(
        self,
        model: nn.Module,
        minibatch_size=1,
        microbatch_size=1,
    ):
        # Ensure user is registered
        user_id = self.contract.functions.getUserId(self.key_hash).call()

        if user_id < 1:
            self.debug_print(f"user not registered on smart contract")
            return False

        # Total memory for job
        model_memory = estimate_memory(model)

        # # Split up into 4 modules (default fragmentation)
        # if max_module_size is None:
        #     max_module_size = model_memory // 1

        # Get model distribution schematic (submoduleId-to-workerIndex)
        config = self.parse_model(model, 4e9)
        distribution = []
        for mod_id, mod_info in config.items():
            if mod_info["type"] == "offloaded":
                distribution.append((mod_info["mod_id"], mod_info["size"]))

        # Update job request struct.
        capacity = sum(d[1] for d in distribution)

        # Number of pipelines
        assert (
            not minibatch_size % microbatch_size,
            "Minibatch must be divisible by microbatch size!",
        )
        n_pipelines = minibatch_size // microbatch_size

        capacity *= n_pipelines

        # Publish job request to smart contract
        validator_ids = self.contract.functions.requestJob(1, capacity).call()

        # Connect to seed validators
        validator_hashes = []
        for validator in validator_ids:
            validator_hash = self.contract.functions.validatorHashById(validator).call()
            validator_hashes.append(validator_hash.encode())

            # Try and grab node connection info from dht
            node_info = self.query_routing_table(validator_hash.encode())

            # Delete space for node info if not found and move on to the next validator
            if node_info is None:
                self.delete(validator_hash)
                self.debug_print(
                    f"Could not connect to validator for job initialize, try again."
                )
                return False

            # Connect to the validator's node and exchange information
            connected = self.connect_dht_node(node_info["host"], node_info["port"])

            if not connected:
                self.delete(validator_hash)
                self.debug_print(
                    f"Could not connect to validator for job initialize, try again."
                )
                return False

        # Get validator connections
        validators = [
            n for n in list(self.nodes.values()) if n.node_id in validator_hashes
        ]

        # Create job request
        job_request = {
            "author": self.key_hash.encode(),
            "seed_validators": validator_hashes,
            "dp_factor": n_pipelines,
            "distribution": distribution,
            "capacity": capacity,
            "workers": [{} for _ in range(n_pipelines)],
        }

        # Get unique job id given current parameters
        job_id = hashlib.sha256(pickle.dumps(job_request)).hexdigest()
        job_request["id"] = job_id

        # Create a struct to receive incoming messages from workers
        for i, (module, mem) in enumerate(distribution):
            self.node_requests[tuple(module)] = []

        # Send job request to multiple validators (seed validators)
        job_req_threads = []
        for validator in validators:
            # t = threading.Thread(
            #     target=self.send_job_req, args=(validator, job_request)
            # )
            # t.start()
            # job_req_threads.append(t)
            self.send_job_req(validator, job_request)
            break  # TODO Send to multiple validators

        # for t in job_req_threads:
        #     t.join()

        # Check that we have received all required workers (ie N-offloaded * DP factor)
        dist_model_config = {}
        for pipeline in range(n_pipelines):
            for mod_id, mem in job_request["distribution"]:
                # if len(self.node_requests[tuple(mod_id)]) == job_request["dp_factor"]:
                # pass
                # else:
                #     raise "Not enough recruited workers."
                mod_id = tuple(mod_id)

                # Wait for loading confirmation from worker nodes
                while not self.node_requests[mod_id]:
                    pass

                module, name = access_module(model, list(mod_id))
                dist_model_config[name] = self.node_requests[
                    mod_id
                ].pop()  # TODO Takes the last (most recent model for now, should accomodate all pipelines in the future)

                # Update job with selected worker
                job_request["workers"][pipeline][mod_id] = dist_model_config[name]

        # Send activation message to validators
        for validator in validators:
            self.send_job_status_update(validator, job_request)

        self.job = job_request

        dist_model = DistributedModel(
            model, self, minibatch_size, microbatch_size, config=dist_model_config
        )
        return dist_model

    def send_job_req(self, validator, job_info):
        message = b"JOBREQ" + pickle.dumps(job_info)
        self.send_to_node(validator, message)

    def parse_model(self, model, max_module_size, config={}, ids: list = []):
        """
        Parse model based on some minimum fragment size
        TODO Option for offloading on user, if enabled we must accommodate some layers on the user before ANY offloading
        to workers are done
        """
        named_children = list(model.named_children())

        # Break down model into children
        for i in range(len(named_children)):
            name, children = named_children[i]
            model_mem = estimate_memory(children)
            new_ids = ids + [i]
            module_type = f"{type(children)}".split(".")[-1].split(">")[0][:-1]

            # Accommodate on user can handle the layer
            if self.available_memory > model_mem:
                self.available_memory -= model_mem
                config[tuple(new_ids)] = {
                    "type": "loaded",
                    "module": module_type,
                    "mod_id": new_ids,
                    "size": model_mem,
                }

            # Append module id to offloaded config if meets the minimum size
            elif model_mem < max_module_size:
                config[tuple(new_ids)] = {
                    "type": "offloaded",
                    "module": module_type,
                    "mod_id": new_ids,
                    "size": model_mem,
                }

            # Recursively break down model if too large
            else:
                sub_config = self.parse_model(
                    children, max_module_size, config, new_ids
                )
                config[tuple(new_ids)] = sub_config

        return config

    # def activate_job(self, job_id, workers):
    #     self.send_to_node()

    def send_job_status_update(self, node, job: dict):
        # Update the job state to the overseeing validators
        job_bytes = b"JOBUPDATE" + pickle.dumps(job)
        self.send_to_node(node, job_bytes)
