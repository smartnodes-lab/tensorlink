from tensorlink.p2p.connection import Connection
from tensorlink.p2p.torch_node import TorchNode

from dotenv import get_key
import hashlib
import json
import logging
import queue
import random
import threading
import time


class User(TorchNode):
    def __init__(
        self,
        request_queue,
        response_queue,
        print_level=logging.DEBUG,
        max_connections: int = 0,
        upnp=True,
        off_chain_test=False,
        local_test=False,
    ):
        super(User, self).__init__(
            request_queue,
            response_queue,
            "U",
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
            local_test=local_test,
        )
        self.print_level = print_level
        self.distributed_graph = {}
        self.worker_stats = {}

        self.debug_print(
            f"Launching User: {self.rsa_key_hash} ({self.host}:{self.port})",
            level=logging.INFO,
        )

        # self.endpoint = create_endpoint(self)
        # self.endpoint_thread = threading.Thread(
        #     target=self.endpoint.run, args=("127.0.0.1", 5029), daemon=True
        # )
        # self.endpoint_thread.start()

        # Some sort of user verification/sign-in could be implemented
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

        if self.off_chain_test is False:
            self.public_key = get_key(".tensorlink.env", "PUBLIC_KEY")
            if not self.public_key:
                self.debug_print(
                    "Public key not found in .env file, using donation wallet..."
                )
                self.public_key = "0x1Bc3a15dfFa205AA24F6386D959334ac1BF27336"

            self.store_value(hashlib.sha256(b"ADDRESS").hexdigest(), self.public_key)

            if self.local_test is False:
                attempts = 0

                self.debug_print("Bootstrapping...")
                while attempts < 3 and len(self.validators) == 0:
                    self.bootstrap()
                    if len(self.validators) == 0:
                        time.sleep(15)
                        self.debug_print("No validators found, trying again...")
                        attempts += 1

                if len(self.validators) == 0:
                    self.debug_print(
                        "No validators found, shutting down...", level=logging.CRITICAL
                    )
                    self.stop()
                    self.terminate_flag.set()

    def handle_data(self, data: bytes, node: Connection) -> bool:
        """
        Callback function to receive streamed data from worker roles.
        """
        try:
            handled = super().handle_data(data, node)
            ghost = 0

            if not handled:
                # We have received a job accept response from a validator handle
                if b"ACCEPT-JOB" == data[:10]:
                    # Start a new thread to handle the job acceptance logic
                    threading.Thread(
                        target=self.finalize_job_creation,
                        args=(data, node),
                        daemon=True,
                    ).start()
                elif b"DECLINE-JOB" == data[:11]:
                    if node.node_id in self.jobs[-1]["seed_validators"]:
                        reason = data[11:]
                        self.debug_print(
                            f"User -> Validator ({node.node_id}) declined job! Reason: {reason}",
                            colour="bright_red",
                            level=logging.ERROR,
                        )
                        self.stop()

                    else:
                        ghost += 1
                elif b"WORKERS" == data[:7]:
                    self.debug_print(f"User -> Received workers from: {node.node_id}")
                    workers = json.loads(data[7:])
                    for worker, stats in workers.items():
                        self.worker_stats[worker] = stats
                else:
                    ghost += 1

            if ghost > 0:
                node.ghosts += ghost
                # TODO: potentially some form of reporting mechanism via ip and port

            return True

        except Exception as e:
            self.debug_print(
                f"User -> error handling data {e}",
                colour="bright_red",
                level=logging.ERROR,
            )
            raise e

    def handle_requests(self, req=None):
        """Handles interactions between model and roles processes"""
        if req is None:
            try:
                req = self.request_queue.get(timeout=3)

            except queue.Empty:
                return

        if req["type"] == "request_job":
            assert self.role == "U", "Must be user to request a job!"
            n_pipelines, dp_factor, distribution, training = req["args"]
            dist_config = self.request_job(
                n_pipelines, dp_factor, distribution, training
            )
            self.response_queue.put({"status": "SUCCESS", "return": dist_config})

        elif req["type"] == "request_workers":
            self.request_worker_info()
            time.sleep(1)
            self.response_queue.put({"status": "SUCCESS", "return": None})

        elif req["type"] == "check_workers":
            workers = self.worker_stats
            self.response_queue.put({"status": "SUCCESS", "return": workers})

        else:
            super().handle_requests(req)

    def request_peers(self):
        pass

    def finalize_job_creation(self, data: bytes, node: Connection):
        """
        Handle the job acceptance logic when a validator accepts a job.
        Runs in a separate thread.
        """
        try:
            job_id = self.jobs[-1]
            job_data = self.query_dht(job_id)

            if job_data and node.node_id in job_data["seed_validators"]:
                self.debug_print(
                    f"User -> Validator ({node.node_id}) accepted job!",
                    colour="bright_green",
                    level=logging.INFO,
                )
                job_id = data[10:74].decode()
                job_data = json.loads(data[74:])
                distribution = job_data["distribution"]

                for mod_id, mod_info in distribution.items():
                    if mod_id not in self.modules:
                        self.modules[mod_id] = mod_info.copy()
                        self.modules[mod_id]["workers"] = []

                    for worker in mod_info["workers"]:
                        worker_id, worker_info = worker
                        # for worker, worker_info in _:
                        # Connect to workers for each model
                        connected = self.connect_worker(
                            worker_info["id"],
                            worker_info["host"],
                            worker_info["port"],
                            mod_id,
                        )

                        if connected:
                            self.debug_print(
                                f"User -> Module: {mod_id} has connected to Worker: {worker_info}"
                            )
                            self.modules[mod_id]["workers"].append(worker_info["id"])
                        else:
                            self.debug_print(
                                f"User -> Error connecting Module: {mod_id} to Worker: {worker_info}"
                            )

                # Update DHT with new worker assignments
                self.store_value(job_id, job_data)
                if node.node_id in self.requests:
                    if job_id in self.requests[node.node_id]:
                        self.requests[node.node_id].remove(job_id)
            else:
                self.debug_print(
                    f"User -> Unexpected ACCEPT-JOB from non-seed validator: {node.node_id}",
                    level=logging.WARNING,
                )

        except Exception as e:
            self.debug_print(
                f"User -> handle job accept error: {e}",
                level=logging.CRITICAL,
                colour="bright_red",
            )
            raise e

    def request_job(self, n_pipelines, dp_factor, distribution, training):
        """Request job through smart contract and set up the relevant connections for a distributed model.
        Returns a distributed nn.Module with built-in RPC calls to workers."""
        # Commented out code as user contract interactions will come later
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
        if len(distribution) != 1:
            # The case where we have a custom model with distributed config
            distribution = {
                k: v for k, v in distribution.items() if v["type"] == "offloaded"
            }

            # Update required network capacity TODO must account for batch size
            capacity = (
                sum(distribution[k]["size"] for k in distribution.keys()) * n_pipelines
            )

            for mod_id, mod_info in distribution.items():
                if mod_info["type"] == "offloaded":
                    self.modules[mod_id] = mod_info

            job_request = {
                "author": self.rsa_key_hash,
                "active": True,
                "hosted": False,
                "capacity": capacity,
                "payment": 0,
                "n_pipelines": n_pipelines,
                "dp_factor": dp_factor,
                "distribution": distribution,
                "n_workers": n_pipelines * len(distribution),
                "seed_validators": validator_ids,
            }
        else:
            # The case where we have a huggingface model name for inference
            job_request = {
                "author": self.rsa_key_hash,
                "active": True,
                "hosted": False,
                "training": training,
                "payment": 0,
                "capacity": 0,
                "n_pipelines": 1,
                "dp_factor": 1,
                "distribution": {},
                "n_workers": 0,
                "model_name": distribution.get("model_name"),
                "seed_validators": validator_ids,
            }

        # Get validator connections
        validators = [
            self.nodes[val_hash]
            for val_hash in validator_ids
            if val_hash in self.validators
        ]

        # Get unique job id given current parameters
        job_hash = hashlib.sha256(json.dumps(job_request).encode()).hexdigest()
        job_request["id"] = job_hash
        self.store_value(job_hash, job_request)
        self.jobs.append(job_hash)

        self.debug_print(f"Creating job request: {job_request}")

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

        # Get updated job info
        job_request = self.query_dht(job_hash)
        self.debug_print(f"Received response from validator: {job_request}")
        distribution = job_request["distribution"]

        # Check that we have received all required workers (ie N-offloaded * DP factor)
        dist_model_config = {}
        for mod_id, module in distribution.items():
            # Wait for loading confirmation from worker roles
            worker_info = self.modules[mod_id]
            if len(worker_info["workers"]) < 1:
                self.debug_print(
                    "Network could not find workers for job.",
                    level=logging.INFO,
                    colour="red",
                )
                return

            # worker_id = worker_info["workers"][0]
            # module, name = access_module(model, config[mod_id]["mod_id"])

            # Update job with selected worker
            # TODO Takes the last (most recent model for now, should accommodate all pipelines in the future)
            # job_request["workers"][0][
            #     worker_id
            # ] = mod_id  # TODO 0 hardcoded and must be replaced with n_pipelines
            dist_model_config[mod_id] = module.copy()

            self.modules[mod_id]["forward_queue"] = {}
            self.modules[mod_id]["backward_queue"] = {}

        # TODO Send activation message to validators
        # for validator in validators:
        #     self.send_job_status_update(validator, job_request)

        return dist_model_config

    def send_job_req(self, validator: Connection, job_info):
        """Send a request to a validator to oversee our job"""
        if validator.node_id not in job_info["seed_validators"]:
            raise "Validator not a seed validator"

        message = b"JOB-REQ" + json.dumps(job_info).encode()
        self._store_request(validator.node_id, job_info["id"])
        self.send_to_node(validator, message)
        start_time = time.time()

        # Wait for validator request and accept timeouts
        while job_info["id"] in self.requests[validator.node_id]:
            if time.time() - start_time > 120:
                # TODO handle validator not responding and request new seed validator thru other seed validators
                self.debug_print(
                    "User -> SEED VALIDATOR TIMED OUT WHILE REQUESTING JOB",
                    colour="bright_yellow",
                    level=logging.WARNING,
                )
                return
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
        job_bytes = b"JOB-UPDATE" + json.dumps(job).encode()
        self.send_to_node(node, job_bytes)

    def get_self_info(self):
        data = super().get_self_info()

        if len(self.jobs) > 0:
            job = self.jobs[-1]
            data["job"] = {"capacity": job["id"]}

        return data

    def request_worker_info(self):
        for validator in self.validators:
            node = self.nodes[validator]
            self.send_to_node(node, b"USER-GET-WORKERS")

    def request_new_worker(self, module_id):
        pass

    def run(self):
        # Accept users and back-check history
        # Get proposees from SC and send our state to them
        # If we are the next proposee, accept info from validators and only add info to the final state if there are
        # 2 or more of the identical info
        super().run()

        while not self.terminate_flag.is_set():
            # Handle job oversight, and inspect other jobs (includes job verification and reporting)
            time.sleep(3)

        self.stop()
