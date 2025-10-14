from tensorlink.ml.utils import get_gpu_memory
from tensorlink.p2p.connection import Connection
from tensorlink.p2p.torch_node import Torchnode
from tensorlink.nodes.keeper import Keeper

import torch.nn as nn
from dotenv import get_key
import psutil
import hashlib
import json
import logging
import time


class Worker(Torchnode):
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
        mining_active=None,
        reserved_memory=None,
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
        self.dht.store(hashlib.sha256(b"ADDRESS").hexdigest(), self.public_key)
        self.keeper = Keeper(self)

        self.debug_print(
            f"Launching Worker: {self.rsa_key_hash} ({self.host}:{self.port})",
            level=logging.INFO,
            tag="Worker",
        )
        self.available_gpu_memory = get_gpu_memory()
        self.total_gpu_memory = self.available_gpu_memory
        self.available_ram = psutil.virtual_memory().available
        self.mining_active = mining_active
        self.reserved_memory = reserved_memory

        if self.off_chain_test is False:
            self.public_key = get_key(".tensorlink.env", "PUBLIC_KEY")
            if not self.public_key:
                self.debug_print(
                    "Public key not found in .env file, using donation wallet...",
                    tag="Worker",
                )
                self.public_key = "0x1Bc3a15dfFa205AA24F6386D959334ac1BF27336"

            self.dht.store(hashlib.sha256(b"ADDRESS").hexdigest(), self.public_key)

            # if self.local_test is False:
            #     attempts = 0
            #
            #     self.debug_print("Bootstrapping...", tag="Worker")
            #     while attempts < 3 and len(self.validators) == 0:
            #         self.bootstrap()
            #         if len(self.validators) == 0:
            #             time.sleep(15)
            #             self.debug_print(
            #                 "No validators found, trying again...", tag="Worker"
            #             )
            #             attempts += 1
            #
            #     if len(self.validators) == 0:
            #         self.debug_print(
            #             "No validators found, shutting down...",
            #             level=logging.CRITICAL,
            #             tag="Worker",
            #         )
            #         self.stop()
            #         self.terminate_flag.set()
        self.keeper.load_previous_state()

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
                        f"Received stats request from: {node.node_id}", tag="Worker"
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
                f"stream data exception: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Worker",
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
                self.keeper.clean_node()
                self.clean_port_mappings()
                self.print_status()

            time.sleep(1)
            counter += 1

    def load_distributed_module(self, module: nn.Module, graph: dict = None):
        pass

    # def proof_of_learning(self, dummy_input: torch.Tensor):
    #     proof = {
    #         "node_id": self.name,
    #         "memory": self.available_gpu_memory,
    #         "learning": self.training,
    #         "model": self.model,
    #     }
    #
    #     if self.training:
    #         proof["output"] = handle_output(self.model(dummy_input)).sum()

    def handle_statistics_request(self, callee, additional_context: dict = None):
        """When a validator requests a stats request, return stats"""
        self.available_gpu_memory = get_gpu_memory()

        # If mining is active, report total GPU memory since we'll stop mining on job acceptance
        if self.mining_active is not None and self.mining_active.value:
            self.available_gpu_memory = self.total_gpu_memory

        stats = {
            "id": self.rsa_key_hash,
            "gpu_memory": self.available_gpu_memory,
            "total_gpu_memory": self.total_gpu_memory,
            "role": self.role,
            "training": self.training,
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

    def print_status(self):
        self.print_base_status()
        print("=============================================\n")
