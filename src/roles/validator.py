from src.p2p.torch_node import TorchNode
from src.p2p.connection import Connection

import threading
import hashlib
import pickle
import time
import os


class Validator(TorchNode):
    def __init__(
        self,
        host: str,
        port: int,
        private_key: str,
        debug: bool = False,
        max_connections: int = 0,
        upnp=True,
    ):
        super(Validator, self).__init__(
            host,
            port,
            private_key,
            debug=debug,
            max_connections=max_connections,
            upnp=upnp,
        )

        # Additional attributes specific to the Validator class
        self.job_ids = []
        self.worker_ids = []
        self.role = b"V"

    def stream_data(self, data, node: Connection):
        """
        Callback function to receive streamed data from worker nodes.
        """
        # Process streamed data and trigger validation if necessary
        try:

            handled = super().stream_data(data, node)

            # Try worker-related tags if not found in parent class
            if not handled:

                # Job acceptance from worker
                if b"ACCEPTJOB" == data[:9]:
                    self.debug_print(f"Validator:worker accepted job")
                    module_id = pickle.loads(data[9:])
                    self.node_requests[tuple(module_id)] = node.node_id

                # Job decline from worker
                elif b"DECLINEJOB" == data[:10]:
                    self.debug_print(f"Validator:worker declined job")
                    pass

                # Job creation request from user
                elif b"JOBREQ" == data[:6]:
                    self.debug_print(f"Validator:user requested job")
                    job_req = pickle.loads(data[6:])
                    self.create_job(job_req)
                    return True

                elif b"JOBUPDATE" == data[:9]:
                    self.debug_print(f"Validator:user update job request")
                    self.update_job(data[9:])

                else:
                    return False

            return True

        except Exception as e:
            self.debug_print(f"Validator:stream_data:{e}")
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

        # Query DHT for user id and reputation
        # user = self.query_routing_table(expected_sample_job["id"])

        # Update connected workers stats
        self.request_worker_stats()

        # Request workers to handle job
        recruitment_threads = []
        n_modules = len(modules)
        current_module = modules.pop(0)

        for key_hash, node in self.nodes.items():
            if (
                node.role == b"W" and node.stats["training"] is True
            ):  # Worker is currently active and has the memory
                if node.stats["memory"] >= current_module[1]:
                    worker = self.nodes[key_hash]
                    t = threading.Thread(
                        target=self.send_job_request,
                        args=(worker, current_module[0], current_module[1]),
                    )
                    t.start()
                    recruitment_threads.append(t)

                    if len(modules) > 1:
                        current_module = modules.pop(0)
                    else:
                        break

        for t in recruitment_threads:
            t.join()

        requesting_node = self.nodes[job_data["author"]]
        recruited_workers = []

        # Cycle thru each model and make sure a worker has accepted them
        for n in range(n_modules):
            mod_id = tuple(job_data["distribution"][n][0])
            candidate_node_id = self.node_requests[tuple(mod_id)]
            candidate_node = self.query_routing_table(candidate_node_id)
            recruited_workers.append([mod_id, candidate_node])

        self.send_to_node(
            requesting_node, b"ACCEPTJOB" + pickle.dumps(recruited_workers)
        )

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
        self.store_key_value_pair(job_data["id"].encode(), job_data)

    def send_job_request(self, node, module_id, module_size: int):
        data = pickle.dumps([module_id, module_size])
        data = b"JOBREQ" + data
        module_id = tuple(module_id)
        self.node_requests[module_id] = None
        self.send_to_node(node, data)
        start_time = time.time()

        while self.node_requests[module_id] is None:
            if time.time() - start_time > 5:
                del self.node_requests[module_id]
                break

    def update_job(self, job_bytes: bytes):
        job = pickle.loads(job_bytes)

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
