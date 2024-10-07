import web3

from src.p2p.torch_node import TorchNode
from src.p2p.connection import Connection
from src.crypto.rsa import get_rsa_pub_key

from web3.exceptions import Web3Exception
from collections import Counter
from eth_abi import encode, decode
from dotenv import get_key
import threading
import hashlib
import pickle
import time
import json
import os


def assert_job_req(job_req: dict, node_id):
    required_keys = [
        "author",
        "capacity",
        "active",
        "n_pipelines",
        "dp_factor",
        "distribution",
        "id",
        # "job_number",
        "n_workers",
        "seed_validators",
        "workers",
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
        debug: bool = False,
        max_connections: int = 0,
        upnp=True,
        off_chain_test=False,
    ):
        super(Validator, self).__init__(
            request_queue,
            response_queue,
            debug=debug,
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
        )

        # Additional attributes specific to the Validator class
        self.role = b"V"

        self.rsa_pub_key = get_rsa_pub_key(self.role, True)
        self.rsa_key_hash = hashlib.sha256(self.rsa_pub_key).hexdigest().encode()

        self.debug_colour = "\033[92m"
        self.debug_print(f"Launching Validator: {self.rsa_key_hash} ({self.host}:{self.port})")

        self.worker_memories = {}

        # Params for smart contract state aggregation
        self.last_loaded_job = 0
        self.jobs_to_publish = []
        self.jobs_to_update = []
        self.validators_to_clear = []

        self.proposal_listener = None
        self.execution_listener = None
        self.proposal_candidate = False
        self.current_proposal = 0

        if off_chain_test is False:
            self.public_key = get_key(".env", "PUBLIC_KEY")
            self.store_value(hashlib.sha256(b"ADDRESS").hexdigest().encode(), self.public_key)

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
                    job_id = data[10:74]
                    module_id = data[74:148]

                    if (
                        node.node_id in self.requests
                        and job_id + module_id in self.requests[node.node_id]
                    ):
                        self.debug_print(f"Validator: worker has accepted job!")
                        self.requests[node.node_id].remove(job_id + module_id)

                    else:
                        ghost += 1

                # Job decline from worker
                elif b"DECLINE-JOB" == data[:11]:
                    self.debug_print(f"Validator: worker has declined job!")
                    if (
                        node.node_id in self.requests
                        and b"JOB-REQ" in self.requests[node.node_id]
                    ):
                        self.requests[node.node_id].remove(b"JOB-REQ")
                    else:
                        ghost += 1

                # Job creation request from user
                elif b"JOB-REQ" == data[:7]:
                    self.debug_print(f"Validator: user requested job.")
                    job_req = pickle.loads(data[7:])

                    # Ensure all job data is present
                    if assert_job_req(job_req, node.node_id) is False:
                        ghost += 1
                        # TODO ask the user to send again if id was specified

                    else:
                        # Get author of job listed on SC and confirm job and roles id TODO to be implemented post-alpha
                        node_info = self.query_dht(node.node_id)

                        if node_info and node_info["reputation"] < 50:
                            ghost += 1

                        else:
                            self.create_job(job_req)

                elif b"STATS-RESPONSE" == data[:14]:
                    self.debug_print(f"Received stats from worker: {node.node_id}")

                    if (
                        node.node_id not in self.requests
                        or b"STATS" not in self.requests[node.node_id]
                    ):
                        ghost += 1

                    else:
                        stats = pickle.loads(data[14:])
                        self.requests[node.node_id].remove(b"STATS")
                        self.nodes[node.node_id].stats = stats
                        self.worker_memories[node.node_id] = stats["memory"]

                elif b"JOB-UPDATE" == data[:10]:
                    self.debug_print(f"Validator:user update job request")
                    self.update_job(data[10:])

                elif b"USER-GET-WORKERS" == data[:16]:
                    self.debug_print(f"User requested workers:")
                    self.request_worker_stats()
                    time.sleep(0.5)
                    stats = {}

                    for worker in self.workers:
                        stats[worker] = self.nodes[worker].stats

                    stats = pickle.dumps(stats)
                    self.send_to_node(node, b"WORKERS" + stats)

                else:
                    return False

            if ghost > 0:
                self.update_node_stats(node.node_id, "GHOST")
                # TODO: potentially some form of reporting mechanism via ip and port

            return True

        except Exception as e:
            self.debug_print(f"Validator: stream_data: {e}")
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
        # Method must be invoked by job request from a user
        # We receive a minimum job information data structure from user
        modules = job_data["distribution"].copy()
        job_id = job_data["id"]
        author = job_data["author"]
        n_pipelines = job_data["n_pipelines"]

        assigned_workers = self.check_job_availability(job_data)

        # Job already exists, decline the job
        if not assigned_workers:
            self.decline_job()

        self.store_value(job_id, job_data)

        # Get workers to handle job
        n_modules = len(modules)

        worker_stats = [
            self.nodes[worker_id].stats
            for worker_id in assigned_workers
        ]

        # Cycle thru offloaded modules and send request to workers for offloading
        recruitment_threads = []
        for module_id, module in modules.items():
            for stream in range(n_pipelines):
                t = threading.Thread(
                    target=self.recruit_worker,
                    args=(
                        job_data["author"],
                        job_id,
                        module_id,
                        module["size"],
                    ),
                )
                t.start()
                recruitment_threads.append(t)

        for t in recruitment_threads:
            t.join()

        # Send recruited worker info to user
        requesting_node = self.nodes[job_data["author"]]
        worker_info = [
            (k, self.query_dht(values["worker"]))
            for k, values in job_data["distribution"].items()
        ]

        if len(worker_info) != len(recruitment_threads):
            self.decline_job()

        self.send_to_node(requesting_node, b"ACCEPT-JOB" + job_id + pickle.dumps(worker_info))

        # job = {
        #     "id": b"",  # Job ID hash
        #     "author": b"",  # Author ID hash
        #     "capacity": 0,  # Combined model size
        #     "dp_factor": 0,  # Number of parallel streams
        #     "distribution": {},  # Distribution graph for a single data parallel stream
        #     "loss": [],  # Global (or individual worker) loss + accuracy
        #     "accuracy": [],
        # }

        # Recruit available workers and send them to user?

        # Store job and replicate to other roles
        # self.store_key_value_pair(job_data["id"].encode(), job_data)

    def decline_job(self, ):
        self.send_to_node()

    def recruit_worker(
        self, user_id: bytes, job_id: bytes, module_id: bytes, module_size: int
    ) -> bool:
        data = pickle.dumps([user_id, job_id, module_id, module_size])
        data = b"JOB-REQ" + data
        found = False

        # Cycle through workers finding the closest free mpc to required mpc
        while not found:
            selected_worker = None
            closest_mem_diff = float("inf")

            for worker_id in self.workers:
                worker_stats = self.nodes[worker_id].stats

                if worker_stats:
                    worker_mem = worker_stats["memory"]

                    if worker_mem >= module_size:
                        memory_diff = worker_mem - module_size

                        if memory_diff < closest_mem_diff:
                            closest_mem_diff = memory_diff
                            selected_worker = worker_stats["id"]

            # If we can no longer find a worker with the required mpc
            if not selected_worker:
                return False

            if isinstance(module_id, str):
                module_id = module_id.encode()

            node = self.nodes[selected_worker]
            self.store_request(node.node_id, job_id + module_id)
            self.send_to_node(node, data)

            start_time = time.time()
            not_found = None
            while module_id in self.requests[node.node_id]:
                if time.time() - start_time > 3:
                    self.requests[node.node_id].remove(module_id)
                    not_found = True
                    break

            if not_found:
                continue

            else:
                node.stats["memory"] -= module_size
                job = self.query_dht(job_id)
                job["distribution"][module_id]["worker"] = node.node_id
                return True
        else:
            return False

    def request_worker_stats(self):
        for worker_id in self.workers:
            connection = self.nodes[worker_id]
            message = b"STATS-REQUEST"
            self.send_to_node(connection, message)
            self.store_request(connection.node_id, b"STATS")
            # TODO disconnect workers who do not respond/have not recently responded to request

        time.sleep(1)

    def distribute_job(self):
        """Distribute job to a few other non-seed validators"""
        for validator in self.validators:
            pass
        pass

    def update_job(self, job_bytes: bytes):
        """Update non-seed validators, loss, accuracy, other info"""
        job = pickle.loads(job_bytes)

    def create_worker(self):
        pass

    def update_worker(self):
        pass

    # def share_info(self):
    #     for validator_ids in self.validators:
    #         roles = self.roles[validator_ids]
    #         self.send_

    """For more intensive, paid jobs that are requested directly from SmartnodesCore"""
    def get_jobs(self):
        current_block = self.chain.eth.block_number
        event_filter = self.chain.eth.filter({
            "fromBlock": self.last_loaded_job,
            "toBlock": current_block,
            "address": self.contract_address,
            "topics": [
                self.sno_events["JobRequest"],
                self.sno_events["JobComplete"],
            ]
        })

        self.last_loaded_job = current_block

        events = event_filter.get_all_entries()
        for event in events:
            print(event)

    """Listen for new proposals created on SmartnodesMultiSig"""
    def listen_for_proposals(self):
        # Proposal-specific event listening
        proposal_flag = threading.Event()

        max_proposals = 1
        # TODO: self.multi_sig_contract.functions.getParams() for setting these types of variables on
        #   node startup (init)
        proposals = {}

        while not self.terminate_flag.is_set():
            for i in range(max_proposals):
                if i not in proposals.keys():
                    try:
                        # Start validation in a new thread
                        t = threading.Thread(target=self.validate_proposal, args=(i, proposal_flag))
                        proposals[i] = t
                        t.start()

                    except Exception as e:
                        self.debug_print(f"Error while validating proposal {i}: {e}")
                        time.sleep(30)  # Sleep briefly on error

            if len(proposals) == max_proposals:
                for thread in proposals.values():
                    thread.join()

                proposals.clear()

                self.debug_print(f"All proposals validated, sleeping for 2 minutes...")
                time.sleep(100)

    """Listen for proposal execution events"""
    def listen_for_proposal_executions(self):
        while not self.terminate_flag.is_set():
            try:
                # Fetch state from the contract
                last_proposal_time, next_proposal_id, num_validators, round_validators = self.multi_sig_contract.functions.getState().call()

                # Check if a new proposal round has started
                if next_proposal_id > self.current_proposal:
                    if self.public_key in round_validators:
                        self.create_proposal()
                    else:
                        self.send_state_updates(round_validators)

                    self.debug_print("Handled new proposal round, waiting for next proposal...")

                else:
                    self.debug_print("No new proposals, waiting...")

            except Exception as e:
                self.debug_print(f"Error during proposal execution check: {e}")

            time.sleep(120)  # Sleep for 2 minutes before checking again

    def validate_proposal(self, proposal_num, proposal_flag):
        # TODO if we are the proposal creator (ie a selected validator), automatically cast a vote.
        #  We should also use our proposal data to quickly verify matching data
        self.debug_print(f"Validation started for proposal: {proposal_num}")

        flag = False
        reason = ""

        while not proposal_flag.is_set():
            try:
                function_types, function_data = self.multi_sig_contract.functions.getCurrentProposal(proposal_num).call()

                for function_type, call_data in zip(function_types, function_data):
                    # Deactivate validator call: attempt connection to determine if validator should be deactivated
                    if proposal_flag.is_set():
                        self.debug_print(f"Validation interrupted for proposal {proposal_num}!")
                        return

                    if function_type == 0:
                        node_address = self.chain.to_checksum_address(
                            decode(["address"], call_data)[0]
                        )
                        node_hash = self.contract.functions.getValidatorBytes(node_address).call().hex()
                        node_info = self.query_dht(node_hash)

                        if node_info:
                            node_host, node_port = node_info["host"], node_info["port"]
                            connected = self.connect_node(node_info["id"], node_host, node_port)

                            # Verify the roles is online and in the network
                            if connected:
                                flag = True
                                reason = f"Validator still online."
                                connection = self.nodes[node_info]
                                self.close_connection_socket(connection, "Heart beat complete.")

                        else:
                            flag = True
                            reason = f"Node not found: {node_hash}"

                    # Create job, ensure job is requested on p2p network
                    elif function_type == 1:
                        # TODO ensure the job type is for the src sub-network
                        user_hash, job_hash, capacities = decode(
                            ["bytes32", "bytes32", "uint256[]"], call_data
                        )

                        user_hash = user_hash.hex()
                        job_hash = job_hash.hex()

                        validated = self.validate_job(job_hash, user_id=user_hash, capacities=capacities)

                        if not validated:
                            flag = True
                            reason = f"Invalid job state! {job_hash}"

                    # Complete job, determine status on p2p network
                    elif function_type == 2:
                        job_hash, worker_addresses = decode(
                            ["bytes32", "address[]"], call_data
                        )

                        job_hash = job_hash.hex()

                        validated = self.validate_job(job_hash, active=False)

                        if not validated:
                            flag = True
                            reason = f"Invalid job state! {job_hash}"

                    else:
                        flag = True
                        reason = "Invalid function type."

                    if flag is True:
                        self.debug_print(f"Job validation failed: {reason}")
                        break

                break

            except Web3Exception as e:
                if "Proposal not found!" in str(e):
                    self.debug_print(f"Proposal {proposal_num} not found, sleeping for 30 sec...")
                    time.sleep(30)
                else:
                    raise e

        if not proposal_flag.is_set() and not flag:
            tx = self.multi_sig_contract.functions.approveTransaction(proposal_num).build_transaction({
                "from": self.public_key,
                "nonce": self.chain.eth.get_transaction_count(self.public_key),
                "gas": 6721975,
                "gasPrice": self.chain.eth.gas_price
            })
            signed_tx = self.chain.eth.account.sign_transaction(tx, get_key(".env", "PRIVATE_KEY"))
            tx_hash = self.chain.eth.send_raw_transaction(signed_tx.rawTransaction)
            self.debug_print(f"Proposal {proposal_num} approved! ({tx_hash})")

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

        if self.multi_sig_contract.functions.hasSubmittedProposal(self.public_key).call():
            time.sleep(30)
            return

        self.debug_print(f"Creating proposal...")
        # Proposal creation mode is active, incoming data is stored in self.proposal_events
        self.proposal_candidate = True
        function_types = []
        function_calldata = []

        def check_proposal(value):
            for v in function_calldata:
                if value == v or isinstance(v, list) and value in v:
                    return True
            return False

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
                        self.close_connection_socket(connection, "Heart beat complete.")
                        function_types.append(0)
                        function_calldata.append(encode([
                            ["address"], [node_address]
                        ]))

        # Publishing new jobs to the contract
        for job_id in self.jobs_to_publish:
            job = self.query_dht(job_id)

            if job:
                user_id = job["author"]
                job_hash = bytes.fromhex(job_id)
                user_hash = bytes.fromhex(user_id)

                if not check_proposal(job_hash):
                    function_types.append(1)
                    function_calldata.append(encode([
                        ["bytes32", "bytes32", "uint256[]"],
                        [user_hash, job_hash, [int(1e9), int(1e9)]]
                    ]))

        # Mark jobs as complete on the contract
        for job_id in self.jobs_to_update:
            job = self.query_dht(job_id)

            if job:
                user_id = job["author"]
                job_hash = bytes.fromhex(job_id)
                user_hash = bytes.fromhex(user_id)
                worker_addresses = []

                for worker_id in job["workers"]:
                    worker_info = self.query_dht(worker_id)

                    if worker_info:
                        worker_host, worker_port = worker_info["host"], worker_info["port"]
                        connected = self.connect_node(worker_info["id"], worker_host, worker_port)

                        # Verify the roles is online and in the network
                        if not connected:
                            worker_node = self.nodes[worker_id]
                            worker_address = self.query_node(
                                hashlib.sha256(b"ADDRESS").hexdigest().encode(),
                                worker_node
                            )

                            if worker_address:
                                connection = self.nodes[worker_id]
                                self.close_connection_socket(connection, "Check complete.")
                                worker_addresses.append(worker_address)

                function_types.append(2)
                function_calldata.append(encode([
                    ["bytes32", "address[]"],
                    [job_hash, worker_addresses]
                ]))

        tx = self.multi_sig_contract.functions.createProposal(
            function_types, function_calldata
        ).build_transaction({
            "from": self.public_key,
            "nonce": self.chain.eth.get_transaction_count(self.public_key),
            "gas": 6721975,
            "gasPrice": self.chain.eth.gas_price
        })
        signed_tx = self.chain.eth.account.sign_transaction(tx, get_key(".env", "PRIVATE_KEY"))
        tx_hash = self.chain.eth.send_raw_transaction(signed_tx.rawTransaction)
        self.debug_print(f"Proposal submitted! ({tx_hash})")

    def send_state_updates(self, validators):
        for job in self.jobs:
            pass

    def run(self):
        super().run()

        # self.proposal_listener = threading.Thread(target=self.listen_for_proposals, daemon=True)
        # self.proposal_listener.start()
        #
        # self.execution_listener = threading.Thread(target=self.listen_for_proposal_executions, daemon=True)
        # self.execution_listener.start()

        # Loop for active job and network moderation
        while not self.terminate_flag.is_set():
            try:
                # Handle job oversight, and inspect other jobs (includes job verification and reporting)
                pass
            except KeyboardInterrupt:
                self.terminate_flag.set()
