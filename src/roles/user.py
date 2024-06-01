import time

from src.ml.distributed import DistributedModel
from src.ml.model_analyzer import *
from src.p2p.connection import Connection
from src.p2p.torch_node import TorchNode

from web3.exceptions import ContractLogicError
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
        debug: bool = False,
        max_connections: int = 0,
        upnp=True,
        off_chain_test=False,
        private_key=None,
    ):
        super(User, self).__init__(
            host,
            port,
            debug=debug,
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
        )

        self.role = b"U"
        self.job = None

        if private_key:
            self.account = self.chain.eth.account.from_key(private_key)

        self.chain.eth.default_account = self.account.address

    def handle_data(self, data: bytes, node: Connection) -> bool:
        """
        Callback function to receive streamed data from worker nodes.
        """
        try:
            handled = super().handle_data(data, node)
            ghost = 0

            if not handled:
                # We have received a job accept request from a validator handle
                if b"ACCEPT-JOB" == data[:10]:
                    if node.node_id in self.job["seed_validators"]:
                        self.debug_print(f"Validator: {node.node_id} accepted job!")
                        distribution = pickle.loads(data[10:])

                        for mod_id, worker_info in distribution:
                            # Connect to workers for each model
                            connected = self.connect_worker(
                                worker_info["id"],
                                worker_info["host"],
                                worker_info["port"],
                                mod_id,
                            )

                            if connected:
                                self.modules[mod_id]["workers"].append(
                                    worker_info["id"]
                                )

                    else:
                        ghost += 1
                else:
                    ghost += 1
            if ghost > 0:
                self.update_node_stats(node.node_id, "GHOST")
                # TODO: potentially some form of reporting mechanism via ip and port

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
        max_module_size=4e9,
        handle_layers=True,
    ):
        """Request job through smart contract and set up the relevant connections for a distributed model.
        Returns a distributed nn.Module with built-in RPC calls to workers."""
        assert not self.off_chain_test, "Cannot request job without SC connection."

        # Ensure we (user) are registered
        user_id = self.contract.functions.getUserId(self.rsa_key_hash.decode()).call()
        if user_id < 1:
            self.debug_print(f"request_job: User not registered on smart contract!")
            return None

        # Get model distribution schematic (moduleId-to-workerIndex)
        config = self.parse_model(model, max_module_size, handle_layers=handle_layers)

        # Create distributed modules data structure for job request
        distribution = {}
        for mod_id, mod_info in config.items():
            if mod_info["type"] == "offloaded":
                # Add offloaded module size and id to job info

                distribution[mod_id] = {"size": mod_info["size"]}
                self.modules[mod_id] = mod_info

        # Number of pipelines
        assert (
            not minibatch_size % microbatch_size,
            "Minibatch must be divisible by microbatch size!",
        )
        n_pipelines = minibatch_size // microbatch_size

        # Update required network capacity TODO must account for batch size
        capacity = (
            sum(distribution[k]["size"] for k in distribution.keys()) * n_pipelines
        )

        # Publish job request to smart contract (args: n_seed_validators, requested_capacity), returns validator IDs
        # TODO check if user already has job and switch to that if so, (re initialize a job if failed to start or
        #  disconnected, request data from validators/workers if disconnected and have to reset info.
        job_id = self.contract.functions.jobIdByUser(user_id).call()
        if job_id > 1:
            self.debug_print(
                f"request_job: User has active job, loading active job. Delete job request if this was unintentional!"
            )
        else:
            tx_hash = self.contract.functions.requestJob(1, capacity).transact(
                {"from": self.account.address}
            )
            tx_receipt = self.chain.eth.wait_for_transaction_receipt(tx_hash)

            if tx_receipt.status != 1:
                try:
                    tx = self.chain.eth.get_transaction(tx_hash)
                    tx_input = tx.input
                    revert_reason = self.chain.eth.call(
                        {"to": tx.to, "data": tx_input}, tx.blockNumber
                    )

                except ContractLogicError as e:
                    revert_reason = f"ContractLogicError: {e}"

                except Exception as e:
                    revert_reason = f"Could not fetch revert reason: {e}"

                self.debug_print(f"request_job: Job request reverted; {revert_reason}")

            self.debug_print("request_job: Job requested on Smart Contract!")

        validator_addresses = self.contract.functions.getJobValidators(job_id).call()

        # Connect to seed validators
        validator_hashes = []
        for validator in validator_addresses:
            # TODO reduce the number of smart contract queries, potentially update the mappings and read fns on SC.
            validator_id = self.contract.functions.validatorIdByAddress(
                validator
            ).call()
            validator_hash = self.contract.functions.validatorHashById(
                validator_id
            ).call()

            validator_hash = validator_hash.encode()
            validator_hashes.append(validator_hash)

            # Try and grab node connection info from dht
            node_info = self.query_dht(validator_hash)

            # Delete space for node info if not found and move on to the next validator
            if node_info is None:
                self.delete(validator_hash)
                self.debug_print(
                    f"request_job: Could not connect to validator for job initialize, try again."
                )
                # TODO retry without creating a new job request on SC
                return False

            # Connect to the validator's node and exchange information
            connected = self.connect_node(
                validator_hash, node_info["host"], node_info["port"]
            )

            if not connected:
                self.delete(validator_hash)
                self.debug_print(
                    f"request_job: Could not connect to validator for job initialize, try again."
                )
                return False

        # Get validator connectionsq
        validators = [
            self.nodes[val_hash]
            for val_hash in validator_hashes
            if val_hash in self.validators
        ]

        # Create job request
        job_request = {
            "author": self.rsa_key_hash,
            "capacity": capacity,
            "dp_factor": n_pipelines,
            "distribution": distribution,
            "job_number": job_number,
            "n_workers": n_pipelines * len(distribution),
            "seed_validators": validator_hashes,
            "workers": [{} for _ in range(n_pipelines)],
        }

        # Get unique job id given current parameters
        job_id = hashlib.sha256(pickle.dumps(job_request)).hexdigest().encode()
        job_request["id"] = job_id
        self.job = job_request

        # Send job request to multiple validators (seed validators)
        job_req_threads = []
        for validator in validators[:1]:
            t = threading.Thread(
                target=self.send_job_req, args=(validator, job_request)
            )
            t.start()
            job_req_threads.append(t)
            break  # TODO Send to multiple validators

        for t in job_req_threads:
            t.join()

        # Check that we have received all required workers (ie N-offloaded * DP factor)
        dist_model_config = {}
        while True:
            pass

        for mod_id, module in job_request["distribution"].items():
            # Wait for loading confirmation from worker nodes
            if job_request["distribution"][mod_id]:
                worker_info = job_request["distribution"][mod_id]

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

    def send_job_req(self, validator: Connection, job_info):
        """Send a request to a validator to oversee our job"""
        if validator.node_id not in job_info["seed_validators"]:
            raise "Validator not a seed validator"
        message = b"JOB-REQ" + pickle.dumps(job_info)
        self.store_request(validator.node_id, job_info["id"])
        self.send_to_node(validator, message)
        start_time = time.time()

        while job_info["id"] in self.requests[validator.node_id]:
            if time.time() - start_time > 60:
                # TODO handle validator not responding and request new seed validator thru other seed validators
                self.debug_print("SEED VALIDATOR TIMED OUT WHILE REQUESTING JOB")
                return

        self.debug_print("")
        return

    def parse_model(
        self,
        model,
        max_module_size,
        config: dict = None,
        ids: list = None,
        handle_layers=False,
        handled_layer=False,
    ) -> dict:
        """
        Parse model based on some minimum submodule size and return a config file containing the
        distributed model configuration.
        # TODO update memory of semi-offloaded modules
        """
        if config is None:
            config = {}
        if ids is None:
            ids = []

        # Create offloaded module data structure for config file
        def create_offloaded(module: nn.Module, module_index: list, module_size: int):
            module_id = (
                hashlib.sha256(str(random.random()).encode()).hexdigest().encode()
            )
            data = {
                "type": "offloaded",
                "module": f"{type(module)}".split(".")[-1].split(">")[0][
                    :-1
                ],  # class name
                "mod_id": module_index,
                "size": module_size,
                "workers": [],
            }
            return module_id, data

        # Create user-loaded module data structure for config file
        def create_loaded(module: nn.Module, module_index: list, module_size: int):
            module_id = (
                hashlib.sha256(str(random.random()).encode()).hexdigest().encode()
            )
            data = {
                "type": "loaded",
                "module": f"{type(module)}".split(".")[-1].split(">")[0][
                    :-1
                ],  # class name
                "mod_id": module_index,
                "size": module_size,
                "workers": [],
            }
            return module_id, data

        named_children = list(model.named_children())
        model_size = estimate_memory(model)

        # If we do not want to handle initial layers and model can fit on worker
        if handle_layers is False and model_size <= max_module_size:
            k, v = create_offloaded(model, [-1], model_size)
            config[k] = v

        # Break first model into children
        for i in range(len(named_children)):
            # Unpack module info
            name, submodule = named_children[i]
            module_memory = estimate_memory(submodule)

            # Update current module id
            new_ids = ids + [i]

            module_type = f"{type(submodule)}".split(".")[-1].split(">")[0][:-1]

            # Try to handle on user if specified
            if handle_layers:
                # If user can handle the layer
                if self.available_memory >= module_memory:
                    self.available_memory -= module_memory
                    k, v = create_loaded(submodule, new_ids, module_memory)
                    config[k] = v
                    handled_layer = True
                    continue

                # Break down model further if we haven't handled first layer
                elif handled_layer is False:
                    sub_config = self.parse_model(
                        submodule,
                        max_module_size,
                        config,
                        new_ids,
                        True,
                        False,
                    )
                    k, v = create_loaded(submodule, new_ids, module_memory)
                    v["subconfig"] = sub_config
                    config[k] = v
                    continue

            # Append module id to offloaded config if meets the minimum size
            if module_memory <= max_module_size:
                k, v = create_offloaded(submodule, new_ids, module_memory)
                config[k] = v

            # Recursively break down model if too large
            else:
                sub_config = self.parse_model(
                    submodule, max_module_size, config, new_ids, True, True
                )
                k, v = create_loaded(submodule, new_ids, module_memory)
                v["subconfig"] = sub_config
                config[k] = v

        return config

    def connect_worker(
        self,
        id_hash: bytes,
        host: str,
        port: int,
        module_id: bytes,
        reconnect: bool = False,
    ) -> bool:
        connected = self.connect_node(id_hash, host, port, reconnect)
        return connected

    # def activate_job(self, job_id, workers):
    #     self.send_to_node()

    def send_job_status_update(self, node, job: dict):
        # Update the job state to the overseeing validators
        job_bytes = b"JOB-UPDATE" + pickle.dumps(job)
        self.send_to_node(node, job_bytes)
