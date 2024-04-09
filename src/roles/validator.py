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
        wallet_address: str,
        debug: bool = False,
        max_connections: int = 0,
    ):
        super(Validator, self).__init__(
            host,
            port,
            wallet_address,
            debug=debug,
            max_connections=max_connections,
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
                    module_id = data[9:]
                    self.node_requests[module_id].append(node.node_id)

                # Job decline from worker
                elif b"DECLINEJOB" == data[:10]:
                    pass

                # Job creation request from user
                elif b"REQJOB" == data[:6]:
                    expected_sample_job = {
                        "id": hashlib.sha256(os.urandom(256)).hexdigest(),
                        "author": self.key_hash,
                        "capacity": 4e9,
                        "dp_factor": 3,
                        "distribution": [1e9, 1e9, 1e9, 1e9],
                    }

                    self.create_job(expected_sample_job)

            else:

                return True

        except Exception as e:
            self.debug_print(f"worker:stream_data:{e}")
            raise e

    def validate(self, data):
        """
        Perform validation by comparing computations with worker nodes.
        """
        # Perform computations using the provided data
        # Compare results with computations from worker nodes
        # Store validation results in self.validation_results
        pass

    def create_job(self, job_data):
        # Method must be invoked by job request from a user
        # We receive a minimum job information data structure from user

        modules = job_data["distribution"].copy()
        self.store_key_value_pair(job_data["id"].encode(), job_data)

        # Query DHT for user id and reputation
        # user = self.query_routing_table(expected_sample_job["id"])

        # Update connected workers stats
        self.request_worker_stats()
        recruitment_threads = []
        n_modules = len(modules)
        current_module = modules.pop(0)

        for key_hash, stats in self.node_stats:
            if (
                stats["training"] is True and stats["role"] == 0
            ):  # Worker is currently active and has the memory
                if stats["memory"] >= current_module:
                    worker = self.routing_table[key_hash]
                    t = threading.Thread(
                        target=self.send_job_request,
                        args=(worker, n_modules - len(modules) + 1, current_module),
                    )
                    t.start()
                    recruitment_threads.append(t)

                    if len(modules) > 1:
                        current_module = modules.pop(0)
                    else:
                        break

        for t in recruitment_threads:
            t.join()

        # Cycle thru each model and make sure a worker has accepted them
        for n in range(n_modules):
            val = self.node_requests[n]
            if isinstance(val, list):
                candidate_node_id = val.pop(0)
                candidate_node = self.query_routing_table(candidate_node_id)
                if isinstance(candidate_node, Connection):
                    self.send_to_node()

        job = {
            "id": b"",  # Job ID hash
            "author": b"",  # Author ID hash
            "capacity": 0,  # Combined model size
            "dp_factor": 0,  # Number of parallel streams
            "distribution": {},  # Distribution graph for a single data parallel stream
            "loss": [],  # Global (or individual worker) loss + accuracy
            "accuracy": [],
        }

        # Recruit available workers and send them to user?

        # Store job and replicate to other nodes
        self.store_key_value_pair(job["id"], job)

    def send_job_request(self, node, module_id, module_size: int):
        data = pickle.dumps([module_id, module_size])
        data = b"JOB" + data
        self.send_to_node(node, data)
        self.node_requests[module_id] = []
        start_time = time.time()

        while not self.node_requests[module_id]:
            if time.time() - start_time > 5:
                self.node_requests[module_id] = None
                break
