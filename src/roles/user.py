from src.ml.distributed import DistributedModel
from src.ml.model_analyzer import *
from src.p2p.connection import Connection
from src.p2p.torch_node import TorchNode
from src.p2p.node_api import *
from src.cryptography.rsa import get_rsa_pub_key

from web3.exceptions import ContractLogicError
import torch.nn as nn
import threading
import requests
import hashlib
import pickle
import random
import time
import os


class User(TorchNode):
    def __init__(
        self,
        debug: bool = False,
        max_connections: int = 0,
        upnp=True,
        off_chain_test=False,
        private_key=None,
    ):
        super(User, self).__init__(
            debug=debug,
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
        )

        self.role = b"U"
        self.distributed_graph = {}

        self.rsa_pub_key = get_rsa_pub_key(self.role, True)
        self.rsa_key_hash = hashlib.sha256(self.rsa_pub_key).hexdigest().encode()

        self.debug_colour = "\033[96m"
        self.debug_print(f"Launching User: {self.rsa_key_hash} ({self.host}:{self.port})")

        self.endpoint = create_endpoint(self)
        self.endpoint_thread = threading.Thread(
            target=self.endpoint.run, args=("127.0.0.1", 5029), daemon=True
        )
        self.endpoint_thread.start()

        # user_id = self.contract.functions.userIdByHash(
        #     self.rsa_key_hash.decode()
        # ).call()
        # if user_id <= 0:
        #     print(
        #         f"User not registered on Smart Nodes! Your current public ID is {self.rsa_key_hash}"
        #     )
        #     print(f"Awaiting user ({self.rsa_key_hash}) registration...")
        #
        #     time.sleep(10)
        #     while (
        #         self.contract.functions.userIdByHash(self.rsa_key_hash.decode()).call()
        #         <= 0
        #     ):
        #         time.sleep(5)

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
                    if node.node_id in self.jobs[-1]["seed_validators"]:
                        self.debug_print(f"Validator ({node.node_id}) accepted job!")
                        job_id = data[10:74]
                        distribution = pickle.loads(data[74:])

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
                                self.requests[node.node_id].remove(job_id)
                    else:
                        ghost += 1
                else:
                    ghost += 1
            if ghost > 0:
                self.update_node_stats(node.node_id, "GHOST")
                # TODO: potentially some form of reporting mechanism via ip and port

            return True

        except Exception as e:
            self.debug_print(f"stream_data:{e}")
            raise e

    def handle_requests(self, req=None):
        if req is None:
            req = self.request_queue.get()

        if req["type"] == "request_job":
            assert self.role == b"U", "Must be user to request a job!"
            n_pipelines, capacity = req["args"]
            self.request_job(capacity, n_pipelines)
        else:
            super().handle_requests(req)

    def request_peers(self):
        pass

    def request_job(
        self,
        capacity,
        n_pipelines
    ):
        """Request job through smart contract and set up the relevant connections for a distributed model.
        Returns a distributed nn.Module with built-in RPC calls to workers."""
        # Commented out code as user contract interactions will come later
        # assert not self.off_chain_test, "Cannot request job without SC connection."
        # # Ensure we (user) are registered
        # user_id = self.contract.functions.userIdByHash(
        #     self.rsa_key_hash.decode()
        # ).call()
        #
        # if user_id < 1:
        #     self.debug_print(f"request_job: User not registered on smart contract!")
        #     return None
        # Publish job request to smart contract (args: n_seed_validators, requested_capacity), returns validator IDs
        # TODO check if user already has job and switch to that if so, (re initialize a job if failed to start or
        #  disconnected, request data from validators/workers if disconnected and have to reset info.
        # job_id = self.contract.functions.jobIdByUser(user_id).call()
        # if job_id >= 1:
        #     self.debug_print(
        #         f"request_job: User has active job, loading active job. Delete job request if this was unintentional!"
        #     )
        # else:
        #     tx_hash = self.contract.functions.requestJob(1, capacity).transact(
        #         {"from": self.account.address}
        #     )
        #     tx_receipt = self.chain.eth.wait_for_transaction_receipt(tx_hash)
        #
        #     if tx_receipt.status != 1:
        #         try:
        #             tx = self.chain.eth.get_transaction(tx_hash)
        #             tx_input = tx.input
        #             revert_reason = self.chain.eth.call(
        #                 {"to": tx.to, "data": tx_input}, tx.blockNumber
        #             )
        #
        #         except ContractLogicError as e:
        #             revert_reason = f"ContractLogicError: {e}"
        #
        #         except Exception as e:
        #             revert_reason = f"Could not fetch revert reason: {e}"
        #
        #         self.debug_print(f"request_job: Job request reverted; {revert_reason}")
        #
        # self.debug_print("request_job: Job requested on Smart Contract!")
        # validator_ids = self.contract.functions.getJobValidators(job_id).call()

        validator_ids = [random.choice(self.validators)]

        # Connect to seed validators
        for validator_id in validator_ids:
            # Try and grab node connection info from dht
            node_info = self.query_dht(validator_id)

            # Delete space for node info if not found and move on to the next validator
            if node_info is None:
                self.delete(validator_id)
                self.debug_print(
                    f"request_job: Could not connect to validator for job initialize, try again."
                )
                # TODO retry without creating a new job request on SC
                return False

            # Connect to the validator's node and exchange information
            connected = self.connect_node(
                validator_id, node_info["host"], node_info["port"]
            )

            if not connected:
                self.delete(validator_id)
                self.debug_print(
                    f"request_job: Could not connect to validator for job initialize, try again."
                )
                return False

        # Get validator connections
        validators = [
            self.nodes[val_hash]
            for val_hash in validator_ids
            if val_hash in self.validators
        ]

        # Create job request
        job_request = {
            "author": self.rsa_key_hash,
            "capacity": capacity,
            "dp_factor": n_pipelines,
            "distribution": distribution,
            # "job_number": self,
            "n_workers": n_pipelines * len(distribution),
            "seed_validators": validator_ids,
            "workers": [{} for _ in range(n_pipelines)],
        }

        # Get unique job id given current parameters
        job_hash = hashlib.sha256(pickle.dumps(job_request)).hexdigest().encode()
        job_request["id"] = job_hash
        self.jobs.append(job_request)

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
        for mod_id, module in distribution.items():
            # Wait for loading confirmation from worker nodes
            worker_info = self.modules[mod_id]
            worker_id = worker_info["workers"][0]
            # module, name = access_module(model, config[mod_id]["mod_id"])

            # Update job with selected worker
            # TODO Takes the last (most recent model for now, should accommodate all pipelines in the future)
            job_request["workers"][0][
                worker_id
            ] = mod_id  # TODO 0 hardcoded and must be replaced with n_pipelines
            dist_model_config[mod_id] = worker_id

        # TODO Send activation message to validators
        # for validator in validators:
        #     self.send_job_status_update(validator, job_request)

        self.jobs[-1] = job_request

        dist_model = DistributedModel(
            model, minibatch_size, microbatch_size, config=dist_model_config
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

        # Wait for validator request and accept timeouts
        while job_info["id"] in self.requests[validator.node_id]:
            if time.time() - start_time > 100:
                # TODO handle validator not responding and request new seed validator thru other seed validators
                self.debug_print("SEED VALIDATOR TIMED OUT WHILE REQUESTING JOB")
                return self.send_job_req(validator, job_info)
        return

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

    def get_self_info(self):
        data = super().get_self_info()

        if len(self.jobs) > 0:
            job = self.jobs[-1]
            data["job"] = {"capacity": job["id"]}

        return data
