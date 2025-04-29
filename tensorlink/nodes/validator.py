from tensorlink.p2p.connection import Connection
from tensorlink.p2p.torch_node import TorchNode
from tensorlink.nodes.contract_manager import ContractManager
from tensorlink.nodes.job_monitor import JobMonitor
from tensorlink.ml.utils import estimate_hf_model_memory
from tensorlink.api.node import create_endpoint, GenerationRequest

from dotenv import get_key
import threading
import hashlib
import logging
import queue
import json
import time
import os


STATE_FILE = "logs/dht_state.json"
LATEST_STATE_FILE = "logs/latest_state.json"


class Validator(TorchNode):
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
        super(Validator, self).__init__(
            request_queue,
            response_queue,
            "V",
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
            local_test=local_test,
        )

        # Additional attributes specific to the Validator class
        self.print_level = print_level

        self.debug_print(
            f"Launching Validator: {self.rsa_key_hash} ({self.host}:{self.port})",
            level=logging.INFO,
        )

        self.worker_memories = {}
        self.all_workers = {}

        # Job monitoring and storage
        self.jobs_to_complete = []
        self.validators_to_clear = []

        # Params for smart contract state aggregation
        self.proposal_flag = threading.Event()
        self.current_proposal = 0

        self.contract_manager = None
        self.proposal_listener = None
        self.execution_listener = None
        self.endpoint = None
        self.endpoint_requests = {"generate": []}

        if off_chain_test is False:
            self.public_key = get_key(".tensorlink.env", "PUBLIC_KEY")
            if self.public_key is None:
                self.debug_print("Public key not found in .env file, terminating...")
                self.terminate_flag.set()

            self.contract_manager = ContractManager(
                self, self.multi_sig_contract, self.chain, self.public_key
            )
            self.store_value(hashlib.sha256(b"ADDRESS").hexdigest(), self.public_key)

            time.sleep(0.1)
            (is_active, pub_key_hash) = self.contract.functions.getValidatorInfo(
                self.public_key
            ).call()

            if is_active and bytes.hex(pub_key_hash) == self.rsa_key_hash:
                self.current_proposal = (
                    self.multi_sig_contract.functions.nextProposalId.call()
                )
                # self.bootstrap()

            else:
                self.debug_print(
                    "Validator is inactive on SmartnodesMultiSig or has a different RSA "
                    f"key [expected: {bytes.hex(pub_key_hash)}, received: {self.rsa_key_hash}).",
                    level=logging.CRITICAL,
                )
                self.terminate_flag.set()

        # Start up the API for handling public jobs
        self.endpoint = create_endpoint(self)
        if not local_test:
            self.add_port_mapping(64747, 64747)

        # Finally, load up previous saved state if any
        self.load_dht_state()

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
                    return self._handle_accept_job(data, node)
                # Job decline from worker
                elif b"DECLINE-JOB" == data[:11]:
                    return self._handle_decline_job(data, node)
                # Job creation request from user
                elif b"JOB-REQ" == data[:7]:
                    return self._handle_job_req(data, node)

                elif b"REQUEST-WORKERS" == data[:15]:
                    return self._handle_request_workers(data, node)

                elif b"ALL-WORKER-STATS" == data[:16]:
                    return self._handle_worker_aggregation_response(data, node)

                elif b"STATS-RESPONSE" == data[:14]:
                    return self._handle_worker_stats_response(data, node)

                elif b"JOB-UPDATE" == data[:10]:
                    self.debug_print(
                        "Validator -> User requested update to job structure"
                    )
                    self.update_job(data[10:])

                elif b"USER-GET-WORKERS" == data[:16]:
                    self.debug_print(
                        "Validator -> User requested workers.", colour="bright_blue"
                    )
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
            self.debug_print(
                f"Validator -> Error handling data: stream_data: {e}",
                colour="bright_red",
                level=logging.ERROR,
            )
            raise e

    def _handle_worker_stats_response(self, data: bytes, node: Connection):
        self.debug_print(
            f"Validator -> Received stats from worker: {node.node_id}: {json.loads(data[14:])}",
            colour="bright_blue",
        )

        if (
            node.node_id not in self.requests
            or b"STATS" not in self.requests[node.node_id]
        ):
            self.debug_print(
                f"Validator -> Received unrequested stats from worker: {node.node_id}",
                colour="red",
                level=logging.WARNING,
            )
            node.ghosts += 1

        else:
            stats = json.loads(data[14:])
            self.requests[node.node_id].remove(b"STATS")
            self.nodes[node.node_id].stats = stats
            self.worker_memories[node.node_id] = stats["gpu_memory"]

    def _handle_worker_aggregation_response(self, data: bytes, node: Connection):
        if (
            node.node_id in self.requests.keys()
            and b"ALL-WORKER-STATS" in self.requests[node.node_id]
        ):
            self.requests[node.node_id].remove(b"ALL-WORKER-STATS")
            workers = json.loads(data[16:])
            # TODO thread for aggregation and worker stats aggregation (ie average/most common values)
            for worker, stats in workers.items():
                self.all_workers[worker] = stats

    def _handle_request_workers(self, data: bytes, node: Connection):
        if node.role == "V":
            t = threading.Thread(
                target=self.request_worker_stats,
                args=(node.node_id,),
                daemon=True,
            )
            t.start()

    def handle_requests(self, request=None):
        """Handles interactions between model and node process"""
        try:
            if request is None:
                try:
                    request = self.request_queue.get(timeout=3)
                except queue.Empty:
                    return
            req_type = request.get("type")
            if not req_type:
                self.response_queue.put(
                    {"status": "FAILURE", "error": "Invalid request type"}
                )
                return

            handlers = {
                "get_jobs": self._handle_get_jobs,
                "send_job_request": self.create_base_job,
                "get_api_request": self._handle_check_api,
                "update_api_request": self._handle_update_api,
            }

            handler = handlers.get(req_type)

            if handler:
                handler(request.get("args", None))
            else:
                super().handle_requests(request)

        except Exception as e:
            self.response_queue.put({"status": "FAILURE", "error": str(e)})

    def _handle_get_jobs(self, request):
        """Check if we have received any job requests for huggingface models and fully hosted or api jobs, then relay
        that information back to the DistributedValidator process"""
        try:
            # Check for API job requests for this node
            if self.rsa_key_hash in self.requests:
                job_req = next(
                    (
                        v
                        for v in self.requests[self.rsa_key_hash]
                        if v.startswith("HF-JOB-REQ")
                    ),
                    None,
                )

                if job_req:
                    # Parse the job data
                    job_data = json.loads(job_req[10:])

                    # Remove the request
                    self._remove_request(self.rsa_key_hash, job_req)

                    # Return with a 'return' key as expected by send_request
                    self.response_queue.put({"status": "SUCCESS", "return": job_data})
                else:
                    # No jobs found
                    self.response_queue.put({"status": "SUCCESS", "return": None})
            else:
                # No requests for this node
                self.response_queue.put({"status": "SUCCESS", "return": None})
        except Exception as e:
            print(e)
            # Handle any unexpected errors
            self.response_queue.put({"status": "FAILURE", "error": str(e)})

    # def _handle_send_job(self, job_data: dict):
    #     distribution = job_data.get("distribution", {})
    #
    #     if distribution:
    #         worker = distribution

    # # Send the updated job data with worker info to the user
    # self.send_to_node(
    #     requesting_node,
    #     b"ACCEPT-JOB" + job_id.encode() + json.dumps(job_data).encode(),
    # )
    #
    # self.jobs.append(job_id)
    #
    # for module, module_info in job_data["distribution"].items():
    #     # Remove worker info and just replace with id
    #     worker_ids = list(a[0] for a in module_info["workers"])
    #     module_info["workers"] = worker_ids
    #
    # job_data["timestamp"] = time.time()
    # job_data["last_seen"] = time.time()
    #
    # self.store_value(job_id, job_data)
    #
    # # Start monitor_job as a background task and store it in the list
    # job_monitor = JobMonitor(self)
    # t = threading.Thread(target=job_monitor.monitor_job, args=(job_id,))
    # t.start()

    def create_hf_job(self, job_info: dict, requesters_ip: str = None):
        # Rate limitation checks for requested jobs
        if requesters_ip:
            if self.rate_limiter.is_blocked(requesters_ip):
                self.debug_print(
                    f"Job declined! Reason: UserIPBlocked ({requesters_ip})"
                )
                return False

            self.rate_limiter.record_attempt(requesters_ip)

        # Huggingface model info checks
        (vram, ram) = estimate_hf_model_memory(
            job_info.get("model_name"), training=False
        )

        if job_info.get("payment", 0) == 0:
            _time = 0.25
        else:
            _time = job_info.get("time")

        job_data = job_info
        job_data["ram"] = ram
        job_data["vram"] = vram
        job_data["time"] = _time

        # Hand off model dissection and worker assignment to DistributedValidator process
        request_value = "HF-JOB-REQ" + json.dumps(job_data)
        self._store_request(self.rsa_key_hash, request_value)

    def _handle_check_api(self, request):
        return_val = None
        if len(self.endpoint_requests["generate"]) > 0:
            return_val = self.endpoint_requests["generate"][0]

        self.response_queue.put({"status": "SUCCESS", "return": return_val})

    def _handle_update_api(self, request: tuple):
        return_val = None
        if self.endpoint_requests["generate"]:
            if len(request) == 2:
                model_name, model_id = request
                api_request: GenerationRequest = self.endpoint_requests["generate"][0]
                if not api_request.processing and api_request.hf_name == model_name:
                    return_val = api_request
                    api_request.processing = True

            elif len(request) == 3:
                model_name, model_id, generated_text = request
                api_request: GenerationRequest = self.endpoint_requests["generate"][0]
                if model_name == api_request.hf_name and api_request.processing:
                    self.endpoint_requests["generate"][0].output = generated_text

        self.response_queue.put({"STATUS": "SUCCESS", "return": return_val})

    def _handle_job_req(self, data: bytes, node: Connection):
        job_req = json.loads(data[7:])
        # Get author of job listed on SC and confirm job and roles id TODO to be implemented post-alpha
        node_info = self.query_dht(node.node_id)

        self.debug_print(
            f"Validator -> User: {node.node_id} requested job -> JobRequest({job_req})",
            colour="bright_blue",
            level=logging.INFO,
        )

        if (
            node.role != "U" or not node_info or node_info["reputation"] < 50
        ):  # TODO reputation
            node.ghosts += 1
        elif job_req.get("model_name"):
            threading.Thread(
                target=self.create_hf_job, args=(job_req, node.host)
            ).start()
        else:
            threading.Thread(target=self.create_base_job, args=(job_req,)).start()

    def _handle_decline_job(self, data: bytes, node: Connection):
        self.debug_print(
            f"Validator -> Worker: {node.node_id} has declined job!",
            colour="red",
            level=logging.INFO,
        )
        if node.node_id in self.requests and b"JOB-REQ" in self.requests[node.node_id]:
            self.requests[node.node_id].remove(b"JOB-REQ")
        else:
            node.ghosts += 1

    def _handle_accept_job(self, data: bytes, node: Connection):
        job_id = data[10:74].decode()
        module_id = data[74:148].decode()

        # Check that we have requested worker for a job
        if (
            node.node_id in self.requests
            and job_id + module_id in self.requests[node.node_id]
        ):
            self.debug_print(
                f"Validator -> Worker: {node.node_id} has accepted job!",
                colour="bright_blue",
                level=logging.INFO,
            )
            self.requests[node.node_id].remove(job_id + module_id)

        else:
            node.ghosts += 1

    def check_job_availability(self, job_data: dict):
        """Asserts that the specified user does not have an active job, and that
        the job capacity can be handled by the network."""
        # job_id = job_data["id"]
        user_id = job_data.get("author")
        capacity = job_data.get("capacity")
        distribution = job_data.get("distribution")

        # Request updated worker statistics
        self.request_worker_stats()

        if user_id and user_id != self.rsa_key_hash:
            # Check that user doesnt have an active job already
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
            self.debug_print(
                f"Validator -> Not enough network capacity for Job:\n"
                f"\tID: {job_data['id']}\n"
                f"\tREQUIRED-MEMORY: {capacity}.\n"
                f"\tNETWORK-MEMORY: {total_memory}\n"
            )
            return False

        # Sort workers by memory in ascending order
        sorted_workers = sorted(
            list(self.worker_memories.items()), key=lambda x: x[1]
        )  # (worker_id, memory)
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
                    sorted_workers = [
                        (
                            (w_id, mem)
                            if w_id != assigned_worker[0]
                            else (w_id, updated_memory)
                        )
                        for w_id, mem in sorted_workers
                    ]
                else:
                    self.debug_print(
                        f"Validator -> No worker found with enough memory for Job:\n"
                        f"\tID: {job_data['id']}\n"
                        f"\tMODULE: {module_id}\n"
                        f"\tREQUIRED-MEMORY: {module_memory}.\n"
                    )
                    return False

        return assigned_workers

    def create_base_job(self, job_data: dict):
        modules, job_id, author, n_pipelines = self._prepare_job(job_data)

        requesting_node = self._get_requesting_node(job_data, author)
        assigned_workers = self.check_job_availability(job_data)

        if not assigned_workers:
            self._decline_job(
                job_data, requesting_node, "Could not find enough workers."
            )
            return

        self.store_value(job_id, job_data)
        worker_connection_info = self._assign_workers_to_modules(
            modules,
            assigned_workers,
            author,
            job_id,
            job_data,
            n_pipelines,
            requesting_node,
        )

        if worker_connection_info is None:
            return  # job declined inside _assign_workers_to_modules

        self._update_job_distribution(
            job_data, worker_connection_info, n_pipelines, requesting_node
        )

        if requesting_node:
            self._send_acceptance(requesting_node, job_id, job_data)
        else:
            self._setup_hosted_job(job_id, job_data)

        self._finalize_job(job_id, job_data)

    def _prepare_job(self, job_data):
        modules = job_data.get("distribution").copy()
        job_id = job_data.get("id")
        author = job_data.get("author")
        n_pipelines = job_data.get("n_pipelines")

        if job_id is None:
            job_id = hashlib.sha256(json.dumps(job_data).encode()).hexdigest()
            job_data["id"] = job_id

        return modules, job_id, author, n_pipelines

    def _get_requesting_node(self, job_data, author):
        if job_data.get("hosted"):
            job_data["author"] = self.rsa_key_hash
            return None
        return self.nodes[author]

    def _decline_job(self, job_data, requesting_node, reason):
        self.debug_print(f"Validator -> Declining job '{job_data['id']}': {reason}")
        self.response_queue.put({"status": "SUCCESS", "return": False})
        if requesting_node:
            self.decline_job(requesting_node)

    def _assign_workers_to_modules(
        self,
        modules,
        assigned_workers,
        author,
        job_id,
        job_data,
        n_pipelines,
        requesting_node,
    ):
        worker_connection_info = {}

        for module_id, module in modules.items():
            worker_assignment = []

            for stream in range(n_pipelines):
                worker_found = False

                for worker_id in assigned_workers:
                    if worker_id not in worker_assignment:
                        if self.recruit_worker(
                            author,
                            job_id,
                            module_id,
                            module["size"],
                            worker_id,
                            module["name"],
                            job_data.get("optimizer", None),
                            job_data.get("training", False),
                        ):
                            worker_assignment.append(worker_id)
                            worker_found = True
                            time.sleep(0.25)
                            break

                if not worker_found:
                    self._decline_job(
                        job_data,
                        requesting_node,
                        "Could not find enough workers for distribution.",
                    )
                    return None

            worker_connection_info[module_id] = [
                (worker_id, self.query_dht(worker_id))
                for worker_id in worker_assignment
            ]

        return worker_connection_info

    def _update_job_distribution(
        self, job_data, worker_connection_info, n_pipelines, requesting_node
    ):
        for module_id, worker_info in worker_connection_info.items():
            job_data["distribution"][module_id]["workers"] = worker_info

        for module_id, module_info in job_data["distribution"].items():
            if len(module_info["workers"]) != n_pipelines:
                self._decline_job(
                    job_data,
                    requesting_node,
                    f"Pipeline initialization error! Expected {n_pipelines}, Received {len(module_info['workers'])}.",
                )
                return None

    def _send_acceptance(self, requesting_node, job_id, job_data):
        self.send_to_node(
            requesting_node,
            b"ACCEPT-JOB" + job_id.encode() + json.dumps(job_data).encode(),
        )

    def _setup_hosted_job(self, job_id, job_data):
        self.debug_print(
            f"Creating public inference job with model {job_data.get('model_name')}"
        )

        for mod_id, module in job_data.get("distribution", {}).items():
            self.modules[mod_id] = {
                "mem_info": mod_id,
                "host": self.rsa_key_hash,
                "forward_queue": {},
                "backward_queue": {},
                "name": module.get("name"),
                "optimizer": None,
                "training": False,
                "workers": module.get("workers", []),
                "distribution": job_data.get("distribution"),
            }
            self.state_updates[mod_id] = []

            if len(self.modules[mod_id]["workers"]) < 1:
                self.debug_print(
                    f"Network could not find workers for job '{job_id}' module {mod_id}.",
                    level=logging.INFO,
                    colour="red",
                )
                self.response_queue.put({"status": "SUCCESS", "return": False})
                return

    def _finalize_job(self, job_id, job_data):
        self.response_queue.put({"status": "SUCCESS", "return": True})

        self.jobs.append(job_id)

        job_data["timestamp"] = time.time()
        job_data["last_seen"] = time.time()

        self.store_value(job_id, job_data)

        job_monitor = JobMonitor(self)
        t = threading.Thread(target=job_monitor.monitor_job, args=(job_id,))
        t.start()

    def decline_job(self, node):
        if node:
            self.send_to_node(node, b"DECLINE-JOB")

    def recruit_worker(
        self,
        user_id: bytes,
        job_id: bytes,
        module_id: bytes,
        module_size: int,
        worker_id: str,
        module_name: str,
        optimizer_name: str = None,
        training: bool = False,
        is_api_job: bool = False,
    ) -> bool:
        if user_id is None:
            user_id = self.rsa_key_hash

        data = json.dumps(
            [
                user_id,
                job_id,
                module_id,
                module_size,
                module_name,
                optimizer_name,
                training,
            ]
        )
        data = b"JOB-REQ" + data.encode()
        node = self.nodes[worker_id]
        self.debug_print(
            f"Validator -> Attempting to recruit worker: '{worker_id}' for job: '{job_id}'"
        )

        # Check worker's available memory
        worker_stats = node.stats
        if worker_stats["gpu_memory"] < module_size:
            self.debug_print(
                f"Validator -> Worker: '{worker_id}' not enough GPU memory"
            )
            return False

        # Send a job request to the worker
        self._store_request(node.node_id, job_id + module_id)
        self.send_to_node(node, data)

        # Await 3 seconds for the job request
        timeout = 3
        start_time = time.time()
        while module_id in self.requests[node.node_id]:
            if time.time() - start_time > timeout:
                self.debug_print(
                    f"Validator -> Worker: '{worker_id}' timed out during recruitment request."
                )
                self.requests[node.node_id].remove(module_id)
                return False

        # Worker accepted the job, update stats
        node.stats["gpu_memory"] -= module_size
        job = self.query_dht(job_id)
        job["distribution"][module_id]["workers"] = node.node_id
        self.debug_print(
            f"Validator -> Worker: '{worker_id}' recruited for job '{job_id}'"
        )
        return True

    def get_workers(self):
        self.all_workers = {}

        self.request_worker_stats()

        for node_id, node in self.nodes.items():
            if node.role == "V":
                self.send_to_node(node, b"REQUEST-WORKERS")
                self._store_request(node_id, "ALL-WORKER-STATS")

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

        time.sleep(2)

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

    # def update_job(self, job_bytes: bytes):
    #     """Update non-seed validators, loss, accuracy, other info"""
    #     job = json.loads(job_bytes)
    #
    # def get_jobs(self):
    #     """For more intensive, paid jobs that are requested directly from SmartnodesCore"""
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
    #
    # def complete_job(self, job_data: dict):
    #     """Decide whether to remove the job or add it to the next state update"""
    #     pass
    #
    # def validate_job(
    #     self,
    #     job_id: bytes,
    #     user_id: bytes = None,
    #     capacities: list = None,
    #     active: bool = None,
    # ) -> bool:
    #     # Grab user and job information
    #     job_info = self.query_dht(job_id)
    #
    #     if job_info:
    #         user_response = None
    #
    #         if user_id:
    #             user_info = self.query_dht(user_id)
    #
    #             if user_info:
    #                 # Connect to user and cross-validate job info
    #                 user_host, user_port = user_info["host"], user_info["port"]
    #                 connected = self.connect_node(user_id, user_host, user_port)
    #
    #                 if connected:
    #                     # Query job information from the user
    #                     user_response = self.query_node(job_id, self.nodes[user_id])
    #
    #         # Query job information from seed validators
    #         job_responses = [
    #             self.query_node(job_id, validator)
    #             for validator in job_info["seed_validators"]
    #         ]
    #
    #         # Include the user's response if it exists
    #         if user_response:
    #             job_responses.append(user_response)
    #
    #         if len(job_responses) > 0:
    #             # Sort workers and seed_validators for comparison
    #             for response in job_responses:
    #                 if isinstance(response, dict):
    #                     response["workers"] = sorted(response["workers"])
    #                     response["seed_validators"] = sorted(
    #                         response["seed_validators"]
    #                     )
    #
    #             # Create hashable tuples for counting most common responses
    #             normalized_responses = [
    #                 (tuple(response["workers"]), tuple(response["seed_validators"]))
    #                 for response in job_responses
    #             ]
    #
    #             # Count the most common response
    #             response_count = Counter(normalized_responses)
    #             most_common, count = response_count.most_common(1)[0]
    #
    #             percent_match = count / len(job_responses)
    #
    #             if percent_match >= 0.66:
    #                 # Gather the full original responses that match the most common normalized response
    #                 matching_responses = [
    #                     response
    #                     for response in job_responses
    #                     if (
    #                         tuple(response["workers"]),
    #                         tuple(response["seed_validators"]),
    #                     )
    #                     == most_common
    #                 ]
    #
    #                 # Validate capacities and state if specified
    #                 if capacities is not None or active is not None:
    #                     for response in matching_responses:
    #                         # Check capacities if provided
    #                         if (
    #                             capacities is not None
    #                             and response.get("capacities") != capacities
    #                         ):
    #                             return False
    #
    #                         # Check state if provided
    #                         if active is not None and response.get("active") != active:
    #                             return False
    #
    #                 # If all checks passed or no capacities/state checks were specified
    #                 return True
    #
    #     return False

    def send_state_updates(self, validators):
        for job in self.jobs:
            pass

    def run(self):
        super().run()

        node_cleaner = threading.Thread(target=self.clean_node, daemon=True)
        node_cleaner.start()

        if self.off_chain_test is False:
            self.execution_listener = threading.Thread(
                target=self.contract_manager.proposal_creator, daemon=True
            )
            self.execution_listener.start()
            time.sleep(15)
            self.proposal_listener = threading.Thread(
                target=self.contract_manager.proposal_validator, daemon=True
            )
            self.proposal_listener.start()

        counter = 0
        # Loop for active job and network moderation
        while not self.terminate_flag.is_set():
            if counter % 300 == 0:
                self.save_dht_state()
            if counter % 120 == 0:
                self.clean_node()
                self.clean_port_mappings()
            if counter % 180 == 0:
                self.print_status()

            time.sleep(1)
            counter += 1

    def stop(self):
        self.save_dht_state()
        super().stop()

    def save_dht_state(self, latest_only=False):
        """
        Serialize and save the DHT state to a file.

        Args:
            latest_only (bool): If True, save only to the latest state file.
                               If False, save to both archive and latest files.
        """
        try:
            # Prepare current state data
            current_data = {
                "workers": {},
                "validators": {},
                "users": {},
                "jobs": {},
                "proposals": {},
                "timestamp": time.time(),  # Add timestamp
            }

            # Collect current state
            for worker_id in self.workers:
                worker = self.query_dht(worker_id)
                current_data["workers"][worker_id] = worker

            for validator_id in self.validators:
                validator = self.query_dht(validator_id)
                current_data["validators"][validator_id] = validator

            for user_id in self.users:
                user = self.query_dht(user_id)
                current_data["users"][user_id] = user

            for job_id in self.jobs:
                job = self.query_dht(job_id)
                current_data["jobs"][job_id] = job

            if self.contract_manager:
                for proposal_id in self.contract_manager.proposals:
                    proposal = self.query_dht(proposal_id)
                    current_data["proposals"][
                        proposal_id
                    ] = proposal  # Fixed missing implementation

            # Save to the latest state file (overwriting previous version)
            with open(LATEST_STATE_FILE, "w") as f:
                json.dump(current_data, f, indent=4)

            # If not latest_only, also save to the archive/permanent state file
            if not latest_only:
                # Load existing archive data if available
                existing_data = {
                    "workers": {},
                    "validators": {},
                    "users": {},
                    "jobs": {},
                    "proposals": {},
                }

                if os.path.exists(STATE_FILE):
                    try:
                        with open(STATE_FILE, "r") as f:
                            existing_data = json.load(f)
                    except json.JSONDecodeError:
                        self.debug_print(
                            "SmartNode -> Existing state file read error.",
                            level=logging.WARNING,
                            colour="red",
                        )

                # Update the archive with current data
                for category in ["workers", "validators", "users", "jobs", "proposals"]:
                    existing_data[category].update(current_data[category])

                # Save updated archive data
                with open(STATE_FILE, "w") as f:
                    json.dump(existing_data, f, indent=4)

            self.debug_print(
                "SmartNode -> DHT state saved successfully to "
                + f"{'both files' if not latest_only else 'latest file only'}.",
                level=logging.INFO,
                colour="green",
            )

        except Exception as e:
            self.debug_print(
                f"SmartNode -> Error saving DHT state: {e}",
                colour="bright_red",
                level=logging.WARNING,
            )

    def load_dht_state(self):
        """Load the DHT state from a file."""
        if os.path.exists(LATEST_STATE_FILE):
            try:
                with open(LATEST_STATE_FILE, "r") as f:
                    state = json.load(f)

                # Restructure state: list only hash and corresponding data
                structured_state = {}
                for category, items in state.items():
                    if category != "timestamp":
                        structured_state[category] = {
                            hash_key: data for hash_key, data in items.items()
                        }
                        self.routing_table.update(items)

                self.debug_print(
                    "SmartNode -> DHT state loaded successfully.", level=logging.INFO
                )

            except Exception as e:
                self.debug_print(
                    f"SmartNode -> Error loading DHT state: {e}",
                    colour="bright_red",
                    level=logging.INFO,
                )
        else:
            self.debug_print(
                "SmartNode -> No DHT state file found.", level=logging.INFO
            )

    def clean_node(self):
        """Periodically clean up node storage"""

        def clean_nodes(nodes):
            nodes_to_remove = []
            for node_id in nodes:
                # Remove any ghost ids in the list
                if node_id not in self.nodes:
                    nodes_to_remove.append(node_id)

                # Remove any terminated connections
                elif self.nodes[node_id].terminate_flag.is_set():
                    role = self.nodes[node_id].role
                    nodes_to_remove.append(node_id)
                    del self.nodes[node_id]

                    if role == "W":
                        del self.all_workers[node_id]

            for node in nodes_to_remove:
                nodes.remove(node)

            # TODO method / request to delete job after certain time or by request of the user.
            #   Perhaps after a job is finished there is a delete request

        clean_nodes(self.workers)
        clean_nodes(self.validators)
        clean_nodes(self.users)

    def print_status(self):
        self.print_base_status()
        print(f" Current Proposal: {self.current_proposal}")
        print("=============================================\n")
