from src.p2p.torch_node import TorchNode
from src.p2p.connection import Connection
from src.crypto.rsa import get_rsa_pub_key

import threading
import hashlib
import pickle
import time
import os


def assert_job_req(job_req: dict):
    required_keys = [
        "author",
        "capacity",
        "dp_factor",
        "distribution",
        "id",
        # "job_number",
        "n_workers",
        "seed_validators",
        "workers",
    ]

    return set(job_req.keys()) == set(required_keys)


class Validator(TorchNode):
    def __init__(
        self,
        request_queue,
        response_queue,
        debug: bool = False,
        max_connections: int = 0,
        upnp=True,
        off_chain_test=False,
        private_key=None,
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
        self.active_jobs = []
        self.last_loaded_job = 0

        if not off_chain_test:
            self.chain.eth.default_account = self.account.address
            if private_key:
                self.account = self.chain.eth.account.from_key(private_key)

    def handle_data(self, data, node: Connection):
        """
        Callback function to receive streamed data from worker nodes.
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
                    if assert_job_req(job_req) is False:
                        ghost += 1
                        # TODO ask the user to send again if id was specified
                    else:
                        # Get author of job listed on SC and confirm job and node id TODO to be implemented post-alpha
                        # job_author = self.contract.functions.userIdByJob(
                        #     job_req["job_number"]
                        # ).call()
                        # job_author_id = self.contract.functions.userKeyById(
                        #     job_author
                        # ).call()
                        # job_id = self.contract.functions.jobIdByUserIdHash(
                        #     node.node_id
                        # ).call()
                        node_info = self.query_dht(node.node_id)

                        if node_info and node_info["reputation"] <= 0:
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

                elif b"JOB-UPDATE" == data[:10]:
                    self.debug_print(f"Validator:user update job request")
                    self.update_job(data[10:])

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
    #     Perform validation by comparing computations with worker nodes.
    #     params:
    #         job_id: job hash
    #         module_id:
    #         epoch:
    #         input: input of module at specific
    #         output:
    #     """
    #     # Perform computations using the provided data
    #     # Compare results with computations from worker nodes
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
    #         # Get worker node ids of specific module in workflow
    #         self.
    #     pass

    def create_job(self, job_data):
        # Method must be invoked by job request from a user
        # We receive a minimum job information data structure from user
        modules = job_data["distribution"].copy()
        job_id = job_data["id"]
        dp_factor = job_data["dp_factor"]
        self.store_value(job_id, job_data)

        # Query SC for user id and reputation?

        # Update connected workers stats
        self.request_worker_stats()
        time.sleep(1)

        # Request workers to handle job
        n_modules = len(modules)
        workers = [
            self.nodes[worker_id].stats
            for worker_id in self.workers
            if self.nodes[worker_id].stats
        ]

        # Cycle thru offloaded modules and send request to workers for offloading
        recruitment_threads = []
        for module_id, module in modules.items():
            for stream in range(dp_factor):
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

        # Store job and replicate to other nodes
        # self.store_key_value_pair(job_data["id"].encode(), job_data)

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
                    worker_mem = worker_stats["mpc"]

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
                node.stats["mpc"] -= module_size
                job = self.query_dht(job_id)
                job["distribution"][module_id]["worker"] = node.node_id
                return True
        else:
            return False

    def confirm_job_integrity(self, job: dict, user: Connection):
        keys = [
            "author",
            "seed_validators",
            "dp_factor",
            "distribution",
            "capacity",
            "workers",
        ]
        assert set(keys) == set(job.keys()), "Invalid received job structure."

        assert job["author"] == user.node_id, "Invalid user."

        assert job["id"] in self.routing_table.keys(), "Job not found in routing table."

        self.routing_table[job["id"]] = job

    def request_worker_stats(self):
        for worker_id in self.workers:
            connection = self.nodes[worker_id]
            message = b"STATS-REQUEST"
            self.send_to_node(connection, message)
            self.store_request(connection.node_id, b"STATS")
            # TODO disconnect workers who do not respond/have not recently responded to request

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

    def share_info(self):
        for validator_ids in self.validators:
            node = self.nodes[validator_ids]
            self.send_

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

    def run(self):
        # Accept users and back-check history
        # Get proposees from SC and send our state to them
        # If we are the next proposee, accept info from validators and only add info to the final state if there are
        # 2 or more of the identical info
        listener = threading.Thread(target=self.listen, daemon=True)
        listener.start()
        
        mp_comms = threading.Thread(target=self.handle_requests, daemon=True)
        mp_comms.start()

        while not self.terminate_flag.is_set():
            # Handle job oversight, and inspect other jobs (includes job verification and reporting)
            # Get active validators listed on SC
            # Get active jobs listed on SC: cross check info between user and validators
            # Submit json to selected proposers/validators
            # Aggregate incoming json if we are selected validator
            pass

        print("Node stopping...")
        for node in self.nodes.values():
            node.stop()

        for node in self.nodes.values():
            node.join()

        listener.join()
        mp_comms.join()

        self.sock.settimeout(None)
        self.sock.close()
        print("Node stopped")
