from tensorlink.p2p.torch_node import TorchNode
from tensorlink.p2p.connection import Connection
from tensorlink.crypto.rsa import get_rsa_pub_key

from web3.exceptions import Web3Exception, ContractLogicError
from collections import Counter
from eth_abi import encode, decode
from dotenv import get_key
import threading
import hashlib
import logging
import time
import json
import os


STATE_FILE = "logs/dht_state.json"


def assert_job_req(job_req: dict, node_id):
    required_keys = [
        "author",
        "capacity",
        "active",
        "n_pipelines",
        "dp_factor",
        "distribution",
        "id",
        "n_workers",
        "seed_validators"
    ]

    if set(job_req.keys()) == set(required_keys):
        if node_id == job_req["author"]:
            return True

    return False


class Validator(TorchNode):
    def __init__(
        self,
        request_queue,
        response_queue,
        print_level=logging.DEBUG,
        max_connections: int = 0,
        upnp=True,
        off_chain_test=False,
        local_test=False
    ):
        super(Validator, self).__init__(
            request_queue,
            response_queue,
            "V",
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
            local_test=local_test
        )

        # Additional attributes specific to the Validator class
        self.print_level = print_level

        self.debug_print(f"Launching Validator: {self.rsa_key_hash} ({self.host}:{self.port})", level=logging.INFO)

        self.worker_memories = {}
        self.all_workers = {}
        self.proposals = {}

        # Job monitoring and storage
        self.active_jobs = {}
        self.jobs_to_delete = []
        self.jobs_to_complete = []
        self.validators_to_clear = []

        # Params for smart contract state aggregation
        self.proposal_flag = threading.Event()
        self.current_proposal = 0

        self.proposal_listener = None
        self.execution_listener = None

        if off_chain_test is False:
            self.public_key = get_key(".env", "PUBLIC_KEY")
            if self.public_key is None:
                self.debug_print("Public key not found in .env file, terminating...")
                self.terminate_flag.set()

            self.store_value(hashlib.sha256(b"ADDRESS").hexdigest(), self.public_key)
            self.id = self.contract.functions.validatorIdByAddress(self.public_key).call()
            if self.id:
                time.sleep(0.1)
                is_active, pub_key_hash, wallet_address = self.contract.functions.getValidatorInfo(
                    self.id
                ).call()

                if is_active and bytes.hex(pub_key_hash) == self.rsa_key_hash:
                    self.current_proposal = self.multi_sig_contract.functions.nextProposalId.call()
                    # self.bootstrap()
                else:
                    self.debug_print("Validator is inactive on SmartnodesMultiSig or has a different RSA "
                                     f"key [expected: {bytes.hex(pub_key_hash)}, received: {self.rsa_key_hash}).",
                                     level=logging.CRITICAL)
                    self.terminate_flag.set()
            else:
                self.debug_print("Validator not listed on SmartnodesMultiSig.", level=logging.CRITICAL)
                self.terminate_flag.set()

    def handle_data(self, data, node: Connection):
        """
        Callback function to receive streamed data from worker roles.
        """
        try:
            handled = super().handle_data(data, node)
            ghost = 0

            # Try worker-related tags if not found in parent class
            if not handled:
                # Job acceptance from worker
                if b"ACCEPT-JOB" == data[:10]:
                    job_id = data[10:74].decode()
                    module_id = data[74:148].decode()

                    # Check that we have requested worker for a job
                    if (
                        node.node_id in self.requests
                        and job_id + module_id in self.requests[node.node_id]
                    ):
                        self.debug_print(f"Validator -> Worker: {node.node_id} has accepted job!",
                                         colour="bright_blue", level=logging.INFO)
                        self.requests[node.node_id].remove(job_id + module_id)

                    else:
                        ghost += 1

                # Job decline from worker
                elif b"DECLINE-JOB" == data[:11]:
                    self.debug_print(f"Validator -> Worker: {node.node_id} has declined job!", colour="red",
                                     level=logging.INFO)
                    if (
                        node.node_id in self.requests
                        and b"JOB-REQ" in self.requests[node.node_id]
                    ):
                        self.requests[node.node_id].remove(b"JOB-REQ")
                    else:
                        ghost += 1

                # Job creation request from user
                elif b"JOB-REQ" == data[:7]:
                    self.debug_print(f"Validator -> User: {node.node_id} requested job.", colour="bright_blue",
                                     level=logging.INFO)
                    job_req = json.loads(data[7:])

                    # Ensure all job data is present
                    if assert_job_req(job_req, node.node_id) is False:
                        self.debug_print(f"Validator -> Declining job: invalid job structure!", colour="red")
                        ghost += 1
                        # TODO ask the user to send again if id was specified

                    else:
                        # Get author of job listed on SC and confirm job and roles id TODO to be implemented post-alpha
                        node_info = self.query_dht(node.node_id)

                        if node.role != "U" or not node_info or node_info["reputation"] < 50:
                            ghost += 1

                        else:
                            threading.Thread(target=self.create_job, args=(job_req,)).start()

                elif b"REQUEST-WORKERS" == data[:15]:
                    if node.role == "V":
                        t = threading.Thread(
                            target=self.request_worker_stats,
                            args=(node.node_id,),
                            daemon=True
                        )
                        t.start()

                elif b"ALL-WORKER-STATS" == data[:16]:
                    if node.node_id in self.requests.keys() and b"ALL-WORKER-STATS" in self.requests[node.node_id]:
                        self.requests[node.node_id].remove(b"ALL-WORKER-STATS")
                        workers = json.loads(data[16:])
                        # TODO thread for aggregation and worker stats aggregation (ie average/most common values)
                        for worker, stats in workers.items():
                            self.all_workers[worker] = stats

                elif b"STATS-RESPONSE" == data[:14]:
                    self.debug_print(f"Validator -> Received stats from worker: {node.node_id}",
                                     colour="bright_blue")

                    if (
                        node.node_id not in self.requests
                        or b"STATS" not in self.requests[node.node_id]
                    ):
                        self.debug_print(
                            f"Validator -> Received unrequested stats from worker: {node.node_id}",
                            colour="red", level=logging.WARNING
                        )
                        ghost += 1

                    else:
                        stats = json.loads(data[14:])
                        self.requests[node.node_id].remove(b"STATS")
                        self.nodes[node.node_id].stats = stats
                        self.worker_memories[node.node_id] = stats["memory"]

                elif b"JOB-UPDATE" == data[:10]:
                    self.debug_print(f"Validator -> User requested update to job structure")
                    self.update_job(data[10:])

                elif b"USER-GET-WORKERS" == data[:16]:
                    self.debug_print(f"Validator -> User requested workers.", colour="bright_blue")
                    self.request_worker_stats()
                    time.sleep(0.5)
                    stats = {}

                    for worker in self.workers:
                        stats[worker] = self.nodes[worker].stats

                    stats = json.dumps(stats)
                    self.send_to_node(node, b"WORKERS" + stats.encode())

                else:
                    return False

            if ghost > 0:
                node.ghosts += ghost
                # TODO: potentially some form of reporting mechanism via ip and port

            return True

        except Exception as e:
            self.debug_print(f"Validator -> Error handling data: stream_data: {e}", colour="bright_red",
                             level=logging.ERROR)
            raise e

    # def validate(self, job_id: bytes, module_id: ):
    #     """
    #     Perform validation by comparing computations with worker roles.
    #     params:
    #         job_id: job hash
    #         module_id:
    #         epoch:
    #         input: input of module at specific
    #         output:
    #     """
    #     # Perform computations using the provided data
    #     # Compare results with computations from worker roles
    #     # Store validation results in self.validation_results
    #
    #     job_info = self.query_routing_table(job_id)
    #
    #     if job_info:
    #         # Confirm job is listed
    #         author = job_info["author"]
    #         listed_author = self.contract.functions.getJobAuthor(author).call()
    #
    #         if author != listed_author:
    #             self.debug_print(f"Invalid/incorrect job")
    #
    #         # Get worker roles ids of specific module in workflow
    #         self.
    #     pass

    def check_job_availability(self, job_data: dict):
        """Asserts that the specified user does not have an active job, and that
        the job capacity can be handled by the network"""
        job_id = job_data["id"]
        user_id = job_data["author"]
        capacity = job_data["capacity"]
        distribution = job_data["distribution"]

        # Request updated worker statistics
        self.request_worker_stats()

        user_info = self.query_dht(user_id, ids_to_exclude=[self.rsa_key_hash])

        # Check for active job
        if user_info:
            current_user_job_id = user_info.get("job")

            if current_user_job_id:
                current_user_job = self.query_dht(current_user_job_id)

                if current_user_job and current_user_job["active"]:
                    return False

        # Check network can handle the requested job
        total_memory = sum(self.worker_memories.values())

        if total_memory < capacity:
            return False

        # Sort workers by memory in ascending order
        sorted_workers = sorted(list(self.worker_memories.items()), key=lambda x: x[1])  # (worker_id, memory)
        assigned_workers = []

        # Iterate over the modules in the distribution
        for module_id, module_info in distribution.items():
            if module_info["type"] == "offloaded":
                module_memory = module_info["size"]
                assigned_worker = None

                # Iterate through the sorted workers and find one with enough memory
                for worker_id, memory in sorted_workers:
                    if memory >= module_memory:
                        # Assign the worker to this module
                        assigned_worker = (worker_id, memory)
                        break

                if assigned_worker:
                    assigned_workers.append(assigned_worker[0])

                    # Update the worker's available memory
                    updated_memory = assigned_worker[1] - module_memory
                    sorted_workers = [(w_id, mem) if w_id != assigned_worker[0] else (w_id, updated_memory)
                                      for w_id, mem in sorted_workers]

        return assigned_workers

    def create_job(self, job_data: dict):
        modules = job_data["distribution"].copy()
        job_id = job_data["id"]
        author = job_data["author"]
        n_pipelines = job_data["n_pipelines"]
        requesting_node = self.nodes[author]

        # Check network availability for job request
        assigned_workers = self.check_job_availability(job_data)

        # If no workers available, decline job
        if not assigned_workers:
            self.send_to_node(requesting_node, b"DECLINE-JOB")
            return

        self.store_value(job_id, job_data)

        # Temporary structure to hold worker connection info
        worker_connection_info = {}

        # Assign workers for each module, ensuring unique workers for the same module across pipelines
        for module_id, module in modules.items():
            worker_assignment = []  # Track assigned workers per pipeline for this module

            for stream in range(n_pipelines):
                worker_found = False

                # Recruit a worker that hasn't been assigned this module in other pipelines
                for worker_id in assigned_workers:
                    worker = self.nodes[worker_id]

                    # Ensure worker isn't already assigned this module in a different pipeline
                    if worker_id not in worker_assignment:
                        if self.recruit_worker(author, job_id, module_id, module["size"], worker_id, module["name"],
                                               module["optimizer"]):
                            worker_assignment.append(worker_id)  # Assign worker to this pipeline's module
                            worker_found = True
                            time.sleep(0.25)
                            break  # Move to the next pipeline for now, multi-worker pipelines coming soon...

                # If no suitable worker found for this pipeline, decline job
                if not worker_found:
                    self.decline_job()
                    return

            # Temporarily store worker connection info for this module
            worker_connection_info[module_id] = [
                (worker_id, self.query_dht(worker_id)) for worker_id in worker_assignment
            ]

        # After recruiting workers, update the original job_data structure
        for module_id, worker_info in worker_connection_info.items():
            job_data["distribution"][module_id]["workers"] = worker_info

        # Check if all modules have the required number of pipelines assigned
        for module, module_info in job_data["distribution"].items():
            if len(module_info["workers"]) != n_pipelines:
                self.decline_job()
                return

        # Send the updated job data with worker info to the user
        self.send_to_node(
            requesting_node,
            b"ACCEPT-JOB" + job_id.encode() + json.dumps(job_data).encode()
        )

        self.jobs.append(job_id)

        for module, module_info in job_data["distribution"].items():
            # Remove worker info and just replace with id
            worker_ids = list(a[0] for a in module_info["workers"])
            module_info["workers"] = worker_ids

        job_data["timestamp"] = time.time()
        job_data["last_seen"] = time.time()

        self.store_value(job_id, job_data)

        # Start monitor_job as a background task and store it in the list
        t = threading.Thread(target=self.monitor_job, args=(job_id,))
        self.active_jobs[job_id] = t
        t.start()

    def decline_job(self):
        pass

    def monitor_job(self, job_id: str):
        """Monitor job progress and workers."""
        self.debug_print(f"Job monitor beginning for job: {job_id}", colour="blue", level=logging.INFO)
        job_data = self.query_dht(job_id)
        online = True

        try:
            while not self.terminate_flag.is_set():
                time.sleep(45)
                self.debug_print(f"Validator -> Inspecting job: {job_id}", colour="blue")

                try:
                    user_data = self.query_dht(job_data["author"])

                    # Check that user is still online
                    connected = self.connect_node(job_data["author"], user_data["host"], user_data["port"])
                    if not connected:
                        online = False

                    if online:
                        # job_data_from_user = self.query_node(job_id, self.nodes[job_data["author"]])
                        # if job_data_from_user is None:
                        #     online = False
                        #
                        # if job_data_from_user is None or not job_data_from_user.get("active"):
                        #     self.debug_print(f"Validator -> Job {job_id} has been marked inactive by user.",
                        #                      colour="red")
                        #     online = False

                        for module_id, module_info in job_data["distribution"].items():
                            if module_info["type"] == "offloaded":
                                for worker in module_info["workers"]:
                                    # TODO Request epoch info from workers
                                    pass

                    # Check if job can be officially marked as done/offline
                    if not online:
                        # Wait for job to be offline for at least 2 minutes to mark as complete
                        if time.time() - job_data["last_seen"] > 60:
                            # Completely delete jobs that went offline within first 3 minutes
                            if time.time() - job_data["timestamp"] < 120:
                                self.debug_print("Validator -> Job timed out during creation.", colour="red")
                            else:
                                self.debug_print("Validator -> Job timed out, marking as complete for Smartnodes...",
                                                 colour="red")
                                job_data["active"] = False
                                self.jobs_to_complete.append(job_id)
                            break

                        self.debug_print("Validator -> User timed out, awaiting user...", colour="red")

                    else:
                        self.debug_print(f"Validator -> Job inspection complete for job: {job_id}", colour="blue")
                        job_data["last_seen"] = time.time()
                        self.routing_table[job_id] = job_data

                except Exception as e:
                    self.debug_print(f"Error monitoring job: {e}", colour="bright_red", level=logging.CRITICAL)
                    break

        finally:
            # Ensure shutdown job is always called at the end
            self.shutdown_job(job_data)

    def shutdown_job(self, job_data):
        del self.active_jobs[job_data["id"]]

        for module_id, module_info in job_data["distribution"].items():
            if module_info["type"] == "offloaded":
                for worker in module_info["workers"]:
                    # TODO Request epoch info from workers
                    try:
                        node = self.nodes[worker]
                        self.send_to_node(node, b"SHUTDOWN-JOB" + module_id.encode())

                    except KeyError:
                        pass

    def recruit_worker(
            self,
            user_id: bytes,
            job_id: bytes,
            module_id: bytes,
            module_size: int,
            worker_id: str,
            module_name: str,
            optimizer_name: str
    ) -> bool:
        data = json.dumps([user_id, job_id, module_id, module_size, module_name, optimizer_name])
        data = b"JOB-REQ" + data.encode()
        node = self.nodes[worker_id]

        # Check worker's available memory
        worker_stats = node.stats
        if worker_stats["memory"] < module_size:
            return False

        self._store_request(node.node_id, job_id + module_id)
        self.send_to_node(node, data)

        start_time = time.time()
        while module_id in self.requests[node.node_id]:
            if time.time() - start_time > 3:
                self.requests[node.node_id].remove(module_id)
                return False

        # Worker accepted the job, update stats
        node.stats["memory"] -= module_size
        job = self.query_dht(job_id)
        job["distribution"][module_id]["workers"] = node.node_id

        return True

    def get_workers(self):
        self.request_worker_stats()

        for node_id, node in self.nodes.items():
            if node.role == "V":
                self.send_to_node(node, b"REQUEST-WORKERS")
                self._store_request(node_id, b"ALL-WORKER-STATS")

        time.sleep(6)

        for worker in self.workers:
            if self.nodes[worker].stats:
                self.all_workers[worker] = self.nodes[worker].stats

    def request_worker_stats(self, send_to=None):
        for worker_id in self.workers:
            connection = self.nodes[worker_id]
            message = b"STATS-REQUEST"
            self.send_to_node(connection, message)
            self._store_request(connection.node_id, b"STATS")
            # TODO disconnect workers who do not respond/have not recently responded to request

        time.sleep(3)

        if send_to is not None:
            node = self.nodes[send_to]
            workers = {}

            for worker in self.workers:
                stats = self.nodes[worker].stats
                if stats:
                    workers[worker] = stats
            self.send_to_node(node, b"ALL-WORKER-STATS" + json.dumps(workers).encode())

    def distribute_job(self):
        """Distribute job to a few other non-seed validators"""
        for validator in self.validators:
            pass
        pass

    def update_job(self, job_bytes: bytes):
        """Update non-seed validators, loss, accuracy, other info"""
        job = json.loads(job_bytes)

    """For more intensive, paid jobs that are requested directly from SmartnodesCore"""
    # def get_jobs(self):
    #     current_block = self.chain.eth.block_number
    #     event_filter = self.chain.eth.filter({
    #         "fromBlock": self.last_loaded_job,
    #         "toBlock": current_block,
    #         "address": self.contract_address,
    #         "topics": [
    #             self.sno_events["JobRequest"],
    #             self.sno_events["JobComplete"],
    #         ]
    #     })
    #
    #     self.last_loaded_job = current_block
    #
    #     events = event_filter.get_all_entries()
    #     for event in events:
    #         print(event)

    def proposal_validator(self):
        """Listen for new proposals created on SmartnodesMultiSig"""

        # Initialize last_execution_block from existing proposals
        while not self.terminate_flag.is_set():
            try:
                # Check if a new round of proposals has started
                current_proposal_id = self.multi_sig_contract.functions.nextProposalId().call()
                time.sleep(3)

                if current_proposal_id != self.current_proposal:
                    self.current_proposal = current_proposal_id
                    self.proposals = {}

                # Get number of proposals we have already found (index for querying next proposal candidate)
                current_proposals = len(self.proposals) + 1
                try:
                    proposal_hash = self.multi_sig_contract.functions.currentProposals(
                        current_proposals - 1).call().hex()
                    time.sleep(3)

                    t = threading.Thread(target=self.validate_proposal,
                                         args=(proposal_hash, current_proposals),
                                         name=f"proposal_validator_{current_proposals}",
                                         daemon=True)
                    self.proposals[proposal_hash] = t
                    t.start()

                    # TODO dynamic based off multisig contract max proposal numbers
                    if current_proposals == 1:
                        self.debug_print(f"Validator -> All proposals validated! Sleeping...",
                                         colour="green", level=logging.DEBUG)
                        time.sleep(300)

                except ContractLogicError:
                    # Proposal has not been published yet, keep waiting
                    time.sleep(60)
                    pass

            except Exception as e:
                self.debug_print(f"Validator -> Error while fetching created proposals: {e}",
                                 colour="bright_red", level=logging.ERROR)

            time.sleep(10)

    def validate_proposal(self, proposal_hash, proposal_num):
        # TODO if we are the proposal creator (ie a selected validator), automatically cast a vote.
        #  We should also use our proposal data to quickly verify matching data
        self.debug_print(f"Validator -> Validation started for proposal: {proposal_hash}",
                         colour="bright_blue", level=logging.INFO)

        # Query network for the detailed proposal info
        proposal_data = self.query_dht(proposal_hash)

        # If no proposal is found
        if proposal_data is None:
            # Perhaps some logic to directly query the node who published the proposal to ensure it is not found TODO
            self.debug_print(f"Validator -> Proposal {proposal_hash} not found in DHT!")
            return

        (
            validators_to_remove,
            job_hashes,
            job_capacities,
            job_workers,
            total_capacity
        ) = proposal_data

        proposal_data_hash = self.hash_proposal_data(
            validators_to_remove,
            job_hashes,
            job_capacities,
            job_workers,
            total_capacity
        ).hex()

        if proposal_data_hash != proposal_hash:
            # Proposal hash must match smart contract listed proposal
            self.debug_print(f"Validator -> Invalid proposal hash!", colour="red")
            return

        # for function_type, call_data in proposal_data:
        #     if self.proposal_flag.is_set():
        #         self.debug_print(f"Validator -> Validation interrupted for proposal {proposal_data_hash}!")
        #         return
        #
        #     # Deactivate validator call: attempt connection to determine if validator should be deactivated
        #     if function_type == 0:
        #         node_address = self.chain.to_checksum_address(
        #             decode(["address"], call_data)[0]
        #         )
        #         node_hash = self.contract.functions.getValidatorBytes(node_address).call().hex()
        #         node_info = self.query_dht(node_hash)
        #
        #         if node_info:
        #             node_host, node_port = node_info["host"], node_info["port"]
        #             connected = self.connect_node(node_info["id"], node_host, node_port)
        #
        #             # Verify the roles is online and in the network
        #             if connected:
        #                 flag = True
        #                 reason = f"Validator still online."
        #                 connection = self.nodes[node_info]
        #                 self.close_connection_socket(connection, "Heart beat complete.")
        #
        #         else:
        #             flag = True
        #             reason = f"Node not found: {node_hash}"
        #
        #     # Create job, ensure job is requested on p2p network
        #     elif function_type == 1:
        #         # TODO ensure the job type is for the tensorlink sub-network
        #         user_hash, job_hash, capacities = decode(
        #             ["bytes32", "bytes32", "uint256[]"], call_data
        #         )
        #
        #         user_hash = user_hash.hex()
        #         job_hash = job_hash.hex()
        #
        #         validated = self.validate_job(job_hash, user_id=user_hash, capacities=capacities)
        #
        #         if not validated:
        #             flag = True
        #             reason = f"Invalid job state! {job_hash}"
        #
        #     # Complete job, determine status on p2p network
        #     elif function_type == 2:
        #         job_hash, worker_addresses = decode(
        #             ["bytes32", "address[]"], call_data
        #         )
        #
        #         job_hash = job_hash.hex()
        #
        #         validated = self.validate_job(job_hash, active=False)
        #
        #         if not validated:
        #             flag = True
        #             reason = f"Invalid job state! {job_hash}"
        #
        #     else:
        #         flag = True
        #         reason = "Invalid function type."
        #
        #     if flag is True:
        #         self.debug_print(f"Validator -> Job validation failed: {reason}", colour="bright_red",
        #         level=logging.WARNING)
        #         break

        try:
            # Determine if proposal can be submitted
            tx = self.multi_sig_contract.functions.approveTransaction(proposal_num).build_transaction({
                "from": self.public_key,
                "nonce": self.chain.eth.get_transaction_count(self.public_key),
                "gas": 6721975,
                "gasPrice": self.chain.eth.gas_price
            })
            signed_tx = self.chain.eth.account.sign_transaction(tx, get_key(".env", "PRIVATE_KEY"))
            tx_hash = self.chain.eth.send_raw_transaction(signed_tx.raw_transaction)
            self.debug_print(f"Validator -> Proposal {proposal_num}: {proposal_hash} approved! ({tx_hash.hex()})", colour="green",
                             level=logging.INFO)

        except Exception as e:
            if "Validator has already voted!" in str(e):
                self.debug_print(f"Validator -> Have already voted on proposal {proposal_num}, continuing...",
                                 colour="green", level=logging.DEBUG)
                pass
            else:
                raise e

    def proposal_creator(self):
        while not self.terminate_flag.is_set():
            try:
                # Fetch state from the contract
                next_proposal_id, round_validators = self.multi_sig_contract.functions.getState().call()
                time.sleep(3)

                if self.public_key in round_validators:
                    # Wait a bit before creating the proposal
                    self.create_proposal()
                    time.sleep(300)

            except Exception as e:
                self.debug_print(f"Validator -> Error processing new entries: {e}",
                                 colour="bright_red", level=logging.ERROR)

            time.sleep(60)

    def validate_job(self, job_id: bytes, user_id: bytes = None, capacities: list = None, active: bool = None) -> bool:
        # Grab user and job information
        job_info = self.query_dht(job_id)

        if job_info:
            user_response = None

            if user_id:
                user_info = self.query_dht(user_id)

                if user_info:
                    # Connect to user and cross-validate job info
                    user_host, user_port = user_info["host"], user_info["port"]
                    connected = self.connect_node(user_id, user_host, user_port)

                    if connected:
                        # Query job information from the user
                        user_response = self.query_node(job_id, self.nodes[user_id])

            # Query job information from seed validators
            job_responses = [
                self.query_node(job_id, validator) for validator in job_info["seed_validators"]
            ]

            # Include the user's response if it exists
            if user_response:
                job_responses.append(user_response)

            if len(job_responses) > 0:
                # Sort workers and seed_validators for comparison
                for response in job_responses:
                    if isinstance(response, dict):
                        response["workers"] = sorted(response["workers"])
                        response["seed_validators"] = sorted(response["seed_validators"])

                # Create hashable tuples for counting most common responses
                normalized_responses = [
                    (tuple(response["workers"]), tuple(response["seed_validators"])) for response in job_responses
                ]

                # Count the most common response
                response_count = Counter(normalized_responses)
                most_common, count = response_count.most_common(1)[0]

                percent_match = count / len(job_responses)

                if percent_match >= 0.66:
                    # Gather the full original responses that match the most common normalized response
                    matching_responses = [
                        response for response in job_responses
                        if (tuple(response["workers"]), tuple(response["seed_validators"])) == most_common
                    ]

                    # Validate capacities and state if specified
                    if capacities is not None or active is not None:
                        for response in matching_responses:
                            # Check capacities if provided
                            if capacities is not None and response.get("capacities") != capacities:
                                return False

                            # Check state if provided
                            if active is not None and response.get("active") != active:
                                return False

                    # If all checks passed or no capacities/state checks were specified
                    return True

        return False

    def create_proposal(self):
        self.debug_print(f"Validator -> Creating proposal...", colour="bright_blue", level=logging.INFO)
        # Proposal creation mode is active, incoming data is stored in self.proposal_events
        validators_to_remove = []
        job_hashes = []
        job_capacities = []
        job_workers = []

        # Removing offline validators from the contract
        for validator in self.validators_to_clear:
            node_info = self.query_dht(validator)
            if node_info:
                node_host, node_port = node_info["host"], node_info["port"]
                connected = self.connect_node(node_info["id"], node_host, node_port)

                # Verify the roles is online and in the network
                if not connected:
                    node_address = self.contract.functions.validatorAddressByHash(validator).call()

                    if node_address:
                        connection = self.nodes[node_info]
                        self.close_connection(connection, "Heart beat complete.")
                        validators_to_remove.append(node_address)

        # Complete jobs on the contract
        for job_id in self.jobs_to_complete:
            job = self.query_dht(job_id)

            if job:
                user_id = job["author"]
                job_hash = bytes.fromhex(job_id)
                capacities = job["capacity"]

                for module_id, module_info in job["distribution"].items():
                    for worker_id in module_info["workers"]:
                        worker_info = self.query_dht(worker_id)

                        if worker_info:
                            worker_host, worker_port = worker_info["host"], worker_info["port"]
                            connected = self.connect_node(worker_info["id"], worker_host, worker_port)

                            # Verify the roles is online and in the network
                            if connected:
                                worker_node = self.nodes[worker_id]
                                worker_address = self.query_node(
                                    hashlib.sha256(b"ADDRESS").hexdigest(),
                                    worker_node
                                )

                                if worker_address:
                                    job_workers.append(worker_address)
                                else:
                                    job_workers.append(self.public_key)

                job_hashes.append(job_hash)
                job_capacities.append(capacities)

        # TODO Temporary fix, we must eventually back check the executed proposal with our jobs, and only delete those
        #  that were included in the proposal

        proposal_hash = self.hash_proposal_data(
            validators_to_remove,
            job_hashes,
            job_capacities,
            job_workers,
            sum(job_capacities)
        )

        proposal = [
            validators_to_remove,
            job_hashes,
            job_capacities,
            job_workers,
            sum(job_capacities)
        ]

        self.store_value(proposal_hash.hex(), proposal)

        # Submit the proposal hash to the SC
        while not self.terminate_flag.is_set():
            # Loop until we submit the proposal or get an error
            try:
                # Test contract call to see if we will revert
                self.multi_sig_contract.functions.createProposal(
                    proposal_hash
                ).call({"from": self.public_key})
                time.sleep(3)

                tx = self.multi_sig_contract.functions.createProposal(
                    proposal_hash
                ).build_transaction({
                    "from": self.public_key,
                    "nonce": self.chain.eth.get_transaction_count(self.public_key),
                    "gas": 6_721_975,
                    "gasPrice": self.chain.eth.gas_price
                })

                signed_tx = self.chain.eth.account.sign_transaction(tx, get_key(".env", "PRIVATE_KEY"))
                tx_hash = self.chain.eth.send_raw_transaction(signed_tx.raw_transaction)
                self.debug_print(f"Validator -> Proposal ({proposal_hash.hex()}) submitted! ({tx_hash.hex()})", colour="green",
                                 level=logging.INFO)
                break

            except Exception as e:
                if "Validator has already submitted a proposal this round!" in str(e):
                    self.debug_print(f"Validator -> Validator has already submitted a proposal "
                                     f"this round!", colour="bright_red", level=logging.INFO)
                    break

                elif "Proposals must be submitted" in str(e):
                    # Proposals must be submitted a certain period after
                    self.debug_print(
                        f"Validator -> createProposal: Not enough time since last proposal! Sleeping...",
                        colour="green", level=logging.DEBUG
                    )
                    return

                else:
                    self.debug_print(f"Validator -> Error creating proposal: {e}", colour="bright_red",
                                     level=logging.INFO)
                    return

        # Wait for next ethereum block
        current_block = self.chain.eth.block_number
        while True:
            new_block = self.chain.eth.block_number
            if new_block > current_block:
                break
            time.sleep(3)

        # Listen for the ProposalReady event in a loop
        while not self.terminate_flag.is_set():
            proposal_id = self.multi_sig_contract.functions.nextProposalId.call()
            time.sleep(3)

            if self.current_proposal != proposal_id:
                # New proposal round detected, scratch current proposal creation
                self.debug_print(f"New proposal round detected, scratching current proposal creation...")
                return
            proposal_number = self.multi_sig_contract.functions.hasSubmittedProposal(self.public_key).call()
            time.sleep(3)
            is_ready = self.multi_sig_contract.functions.isProposalReady(proposal_number).call()
            time.sleep(3)

            if is_ready:
                self.debug_print("Validator -> Proposal ready for execution!",
                                 colour="blue", level=logging.INFO)

                # Execute the proposal
                try:
                    self.multi_sig_contract.functions.executeProposal(
                        validators_to_remove,
                        job_hashes,
                        job_capacities,
                        job_workers,
                        sum(job_capacities)
                    ).call({
                        "from": self.public_key
                    })
                    time.sleep(3)

                    execute_tx = self.multi_sig_contract.functions.executeProposal(
                        validators_to_remove,
                        job_hashes,
                        job_capacities,
                        job_workers,
                        sum(job_capacities)  # TODO execute sum on blockchain
                    ).build_transaction({
                        "from": self.public_key,
                        "nonce": self.chain.eth.get_transaction_count(self.public_key),
                        "gas": 6_721_975,
                        "gasPrice": self.chain.eth.gas_price
                    })

                    signed_execute_tx = self.chain.eth.account.sign_transaction(
                        execute_tx, get_key(".env", "PRIVATE_KEY")
                    )
                    execute_tx_hash = self.chain.eth.send_raw_transaction(
                        signed_execute_tx.raw_transaction
                    )
                    self.debug_print(f"Validator -> Proposal executed! ({execute_tx_hash.hex()})",
                                     colour="green", level=logging.INFO)

                    self.validators_to_clear = []
                    self.jobs_to_complete = []

                    # Return after successful execution
                    return

                except Exception as e:
                    if "Not enough proposal votes!" in str(e):
                        self.debug_print(f"Validator -> Not enough proposal votes, sleeping...",
                                         colour="green", level=logging.DEBUG)

                    else:
                        self.debug_print(f"Validator -> Error executing proposal: {e}",
                                         colour="bright_red", level=logging.ERROR)
                        return
            else:
                self.debug_print("Validator -> Proposal is not ready for execution. sleeping...")

            # Sleep or adjust as needed to reduce polling frequency
            time.sleep(60)

    def send_state_updates(self, validators):
        for job in self.jobs:
            pass

    def hash_proposal_data(
        self,
        validators_to_remove,
        job_hashes,
        job_capacities,
        workers,
        total_capacity
    ):
        validators_to_remove = [self.chain.to_checksum_address(validator) for validator in validators_to_remove]
        workers = [self.chain.to_checksum_address(worker) for worker in workers]
        encoded_data = encode(
            ["address[]", "bytes32[]", "uint256[]", "address[]", "uint256"],
            [
                validators_to_remove,
                job_hashes,
                job_capacities,
                workers,
                total_capacity
            ]
        )

        return self.chain.keccak(encoded_data)

    def run(self):
        super().run()
        node_cleaner = threading.Thread(target=self.clean_node, daemon=True)
        node_cleaner.start()

        if self.off_chain_test is False:
            self.proposal_listener = threading.Thread(target=self.proposal_validator, daemon=True)
            self.proposal_listener.start()

            self.execution_listener = threading.Thread(target=self.proposal_creator, daemon=True)
            self.execution_listener.start()

        # Loop for active job and network moderation
        while not self.terminate_flag.is_set():
            time.sleep(1)

    def stop(self):
        self.save_dht_state()
        super().stop()

    def save_dht_state(self):
        """Serialize and save the DHT state to a file."""
        try:
            # Load existing data if available
            existing_data = {"workers": {}, "validators": {}, "users": {}, "jobs": {}, "proposals": {}}

            if os.path.exists(STATE_FILE):
                try:
                    with open(STATE_FILE, "r") as f:
                        existing_data = json.load(f)
                except json.JSONDecodeError:
                    self.debug_print("SmartNode -> Existing state file read error.", level=logging.WARNING,
                                     colour="red")

            # Append data to each category
            for worker_id in self.workers:
                worker = self.query_dht(worker_id)
                existing_data["workers"][worker_id] = worker

            for validator_id in self.validators:
                validator = self.query_dht(validator_id)
                existing_data["validators"][validator_id] = validator

            for user_id in self.users:
                user = self.query_dht(user_id)
                existing_data["users"][user_id] = user

            for job_id in self.jobs:
                job = self.query_dht(job_id)
                existing_data["jobs"][job_id] = job

            for proposal_id in self.proposals:
                pass

            # Save updated data back to the file
            with open(STATE_FILE, "w") as f:
                json.dump(existing_data, f, indent=4)

            self.debug_print("SmartNode -> DHT state saved successfully.", level=logging.INFO, colour="green")

        except Exception as e:
            self.debug_print(f"SmartNode -> Error saving DHT state: {e}", colour="bright_red",
                             level=logging.WARNING)

    def load_dht_state(self):
        """Load the DHT state from a file."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)

                self.routing_table.update(state)
                self.debug_print("SmartNode -> DHT state loaded successfully.", level=logging.INFO)

            except Exception as e:
                self.debug_print(f"SmartNode -> Error loading DHT state: {e}", colour="bright_red", level=logging.INFO)

    def clean_node(self):
        """Periodically clean up node storage"""
        def clean_nodes(nodes):
            nodes_to_remove = []
            for node_id in nodes:
                # Remove any ghost user_ids in self.users
                if node_id not in self.nodes:
                    nodes_to_remove.append(node_id)

                # Remove any terminated connections
                elif self.nodes[node_id].terminate_flag.is_set():
                    nodes_to_remove.append(node_id)
                    del self.nodes[node_id]

            for node in nodes_to_remove:
                nodes.remove(node)

        while not self.terminate_flag.is_set():
            time.sleep(300)

            self.save_dht_state()

            for job_id in self.jobs:
                job_data = self.query_dht(job_id)

                if job_data["active"] is False:
                    # Remove old jobs (not in jobs to upload to contract or in ones to delete
                    if job_id not in self.jobs_to_complete or job_id in self.jobs_to_delete:
                        self.jobs.remove(job_id)
                        self.__delete(job_id)

            clean_nodes(self.workers)
            clean_nodes(self.validators)
            clean_nodes(self.users)

            # TODO method / request to delete job after certain time or by request of the user.
            #   Perhaps after a job is finished there is a delete request
