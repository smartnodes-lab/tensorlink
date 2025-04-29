from tensorlink.ml.utils import get_gpu_memory, handle_output
from tensorlink.p2p.connection import Connection
from tensorlink.p2p.torch_node import TorchNode

import torch
import torch.nn as nn
from dotenv import get_key
import psutil
import hashlib
import json
import logging
import time
import os


STATE_FILE = "logs/dht_state.json"
LATEST_STATE_FILE = "logs/latest_state.json"


class Worker(TorchNode):
    """
    Todo:
        - link workers to database or download training data for complete offloading
        - different subclasses of Worker for mpc requirements to designate mpc-specific
            tasks, ie distributing a model too large to handle on a single computer / user
    """

    def __init__(
        self,
        request_queue,
        response_queue,
        print_level=logging.INFO,
        max_connections: int = 0,
        upnp=True,
        off_chain_test=False,
        local_test=False,
    ):
        super(Worker, self).__init__(
            request_queue,
            response_queue,
            "W",
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
            local_test=local_test,
        )

        self.training = False
        self.role = "W"
        self.print_level = print_level
        self.loss = None
        self.store_value(hashlib.sha256(b"ADDRESS").hexdigest(), self.public_key)

        self.debug_print(
            f"Launching Worker: {self.rsa_key_hash} ({self.host}:{self.port})",
            level=logging.INFO,
        )
        self.available_gpu_memory = get_gpu_memory()
        self.total_gpu_memory = self.available_gpu_memory
        self.available_ram = psutil.virtual_memory().available

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

    def handle_data(self, data: bytes, node: Connection):
        """
        Handle incoming tensors from connected roles and new job requests
        Todo:
            - ensure correct roles sending data
            - potentially move/forward method directly to Connection to save data via identifying data
                type and relevancy as its streamed in, saves bandwidth if we do not need the data / spam
        """

        try:
            handled = super().handle_data(data, node)
            ghost = 0

            # Try worker-related tags if not found in parent class
            if not handled:
                # Try worker-related tags
                if b"STATS-REQUEST" == data[:13]:
                    self.debug_print(
                        f"Worker -> Received stats request from: {node.node_id}"
                    )
                    self.handle_statistics_request(node)

                elif b"SHUTDOWN-JOB" == data[:12]:
                    if node.role == "V":
                        module_id = data[12:76].decode()
                        self.modules[module_id]["termination"] = True

                elif b"JOB-REQ" == data[:7]:
                    self._handle_job_req(data, node)

                # elif b"PoL" == data[:3]:
                #     self.debug_print(f"RECEIVED PoL REQUEST")
                #     if self.training and self.model:
                #         dummy_input = json.loads(data[3:])
                #
                #         proof_of_learning = self.proof_of_learning(dummy_input)
                #
                #         self.send_to_node(roles, proof_of_learning)

                else:
                    ghost += 1

            if ghost > 0:
                node.ghosts += ghost
                # TODO: potentially some form of reporting mechanism via ip and port
                return False

            return True

        except Exception as e:
            self.debug_print(
                f"Worker -> stream data exception: {e}",
                colour="bright_red",
                level=logging.ERROR,
            )
            raise e

    def _handle_job_req(self, data: bytes, node: Connection):
        try:
            if node.role == "V":
                # Accept job request from validator if we can handle it
                (
                    user_id,
                    job_id,
                    module_id,
                    module_size,
                    module_name,
                    optimizer_name,
                    training,
                ) = json.loads(data[7:])

                if self.available_gpu_memory >= module_size:  # TODO Ensure were active?
                    # Respond to validator that we can accept the job
                    if module_name is None:
                        module_name = ""

                    # Store a request to wait for the user connection
                    self._store_request(user_id, module_id + module_name)

                    if training:
                        self._store_request(user_id, "OPTIMIZER" + optimizer_name)

                    data = b"ACCEPT-JOB" + job_id.encode() + module_id.encode()

                    # Update available memory
                    self.available_gpu_memory -= module_size

                else:
                    data = b"DECLINE-JOB"

            else:
                node.stop()

            self.send_to_node(node, data)

        except Exception as e:
            print(data)
            print(node.main_port)
            raise e

    def run(self):
        # Accept users and back-check history
        # Get proposees from SC and send our state to them
        super().run()

        counter = 0
        while not self.terminate_flag.is_set():
            if counter % 180 == 0:
                self.clean_node()
                self.clean_port_mappings()

            time.sleep(1)
            counter += 1

    def load_distributed_module(self, module: nn.Module, graph: dict = None):
        pass

    def proof_of_learning(self, dummy_input: torch.Tensor):
        proof = {
            "node_id": self.name,
            "memory": self.available_gpu_memory,
            "learning": self.training,
            "model": self.model,
        }

        if self.training:
            proof["output"] = handle_output(self.model(dummy_input)).sum()

    def handle_statistics_request(self, callee, additional_context: dict = None):
        """When a validator requests a stats request, return stats"""
        stats = {
            "id": self.rsa_key_hash,
            "gpu_memory": self.available_gpu_memory,
            "total_gpu_memory": self.total_gpu_memory,
            "role": self.role,
            "training": self.training,
            # "connection": self.connections[i], "latency_matrix": self.connections[i].latency
        }

        if additional_context is not None:
            for k, v in additional_context.items():
                if k not in stats.keys():
                    stats[k] = v

        stats_bytes = json.dumps(stats).encode()
        stats_bytes = b"STATS-RESPONSE" + stats_bytes
        self.send_to_node(callee, stats_bytes)

    def activate(self):
        self.training = True

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
                "timestamp": time.time(),
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
                for category in ["workers", "validators", "users", "jobs"]:
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
                    nodes_to_remove.append(node_id)
                    del self.nodes[node_id]

            for node in nodes_to_remove:
                nodes.remove(node)

        for job_id in self.jobs:
            job_data = self.query_dht(job_id)

            if job_data["active"] is False:
                self.jobs.remove(job_id)
                self._delete_item(job_id)

        clean_nodes(self.workers)
        clean_nodes(self.validators)
        clean_nodes(self.users)

        self.print_status()

    def print_status(self):
        self.print_base_status()
        print("=============================================\n")
