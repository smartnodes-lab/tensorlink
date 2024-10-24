from tensorlink.p2p.torch_node import TorchNode
from tensorlink.p2p.connection import Connection
from tensorlink.ml.utils import estimate_memory, handle_output, get_gpu_memory
from tensorlink.crypto.rsa import get_rsa_pub_key

from dotenv import get_key
import torch.nn as nn
import threading
import hashlib
import pickle
import torch


class Worker(TorchNode):
    """
    Todo:
        - convert pickling to json for security (?)
        - process other jobs/batches while waiting for worker response (?)
        - link workers to database or download training data for complete offloading
        - different subclasses of Worker for mpc requirements to designate mpc-specific
            tasks, ie distributing a model too large to handle on a single computer / user
        - function that detaches the huggingface wrapped outputs tensor without modifying the rest
    """

    def __init__(
        self,
        request_queue,
        response_queue,
        debug: bool = False,
        max_connections: int = 0,
        upnp=True,
        off_chain_test=False,
        local_test=False
    ):
        super(Worker, self).__init__(
            request_queue,
            response_queue,
            debug=debug,
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
            local_test=local_test
        )

        self.training = False
        self.role = b"W"
        self.loss = None
        self.public_key = get_key(".env", "PUBLIC_KEY")
        self.store_value(hashlib.sha256(b"ADDRESS").hexdigest().encode(), self.public_key)

        self.rsa_pub_key = get_rsa_pub_key(self.role, True)
        self.rsa_key_hash = hashlib.sha256(self.rsa_pub_key).hexdigest().encode()

        self.debug_colour = "\033[94m"
        self.debug_print(f"Launching Worker: {self.rsa_key_hash} ({self.host}:{self.port})")
        self.available_memory = 2e9

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
                    self.debug_print(f"Received stats request from: {node.node_id}")
                    self.handle_statistics_request(node)

                elif b"JOB-REQ" == data[:7]:
                    try:
                        # Accept job request from validator if we can handle it
                        user_id, job_id, module_id, module_size = pickle.loads(data[7:])

                        if (
                            self.available_memory >= module_size
                        ):  # TODO Ensure were active?
                            # Respond to validator that we can accept the job
                            # Store a request to wait for the user connection as well
                            self.store_request(user_id + module_id, b"AWAIT-USER")
                            data = b"ACCEPT-JOB" + job_id + module_id

                            # Update available memory
                            self.available_memory -= module_size

                        else:
                            data = b"DECLINE-JOB"

                        self.send_to_node(node, data)

                    except Exception as e:
                        print(data)
                        print(node.main_port)
                        raise e

                elif b"OPTIMIZER" == data[:9]:
                    module_id, optimizer_fn, optimizer_kwargs = pickle.loads(data[9:])
                    self.state_updates[module_id].append((optimizer_fn, optimizer_kwargs))

                # elif b"PoL" == data[:3]:
                #     self.debug_print(f"RECEIVED PoL REQUEST")
                #     if self.training and self.model:
                #         dummy_input = pickle.loads(data[3:])
                #
                #         proof_of_learning = self.proof_of_learning(dummy_input)
                #
                #         self.send_to_node(roles, proof_of_learning)
                #
                # elif b"TENSOR" == data[:6]:
                #     if self.training:
                #         tensor = pickle.loads(data[6:])
                #
                #         # Confirm identity/role of roles
                #         if roles in self.inbound:
                #             self.forward_relays.put(tensor)
                #         elif roles in self.outbound:
                #             self.backward_relays.put(tensor)

                else:
                    ghost += 1

            if ghost > 0:
                self.update_node_stats(node.node_id, "GHOST")
                # TODO: potentially some form of reporting mechanism via ip and port
                return False

            return True

        except Exception as e:
            self.debug_print(f"worker:stream_data:{e}")
            raise e

    def run(self):
        # Accept users and back-check history
        # Get proposees from SC and send our state to them
        super().run()

        while not self.terminate_flag.is_set():
            # Handle job oversight, and inspect other jobs (includes job verification and reporting)
            pass

    def load_distributed_module(self, module: nn.Module, graph: dict = None):
        pass

    def proof_of_learning(self, dummy_input: torch.Tensor):
        proof = {
            "node_id": self.name,
            "memory": self.available_memory,
            "learning": self.training,
            "model": self.model,
        }

        if self.training:
            proof["output"] = handle_output(self.model(dummy_input)).sum()

    def handle_statistics_request(self, callee, additional_context: dict = None):
        """When a validator requests a stats request, return stats"""
        stats = {
            "id": self.rsa_key_hash,
            "memory": self.available_memory,
            "role": self.role,
            "training": self.training,
            # "connection": self.connections[i], "latency_matrix": self.connections[i].latency
        }

        if additional_context is not None:
            for k, v in additional_context.items():
                if k not in stats.keys():
                    stats[k] = v

        stats_bytes = pickle.dumps(stats)
        stats_bytes = b"STATS-RESPONSE" + stats_bytes
        self.send_to_node(callee, stats_bytes)

    def activate(self):
        self.training = True

    """Key Methods to Implement"""
    # def request_worker(self, roles, module_memory: int, module_type: int):
    #     # module_type = 0 when model is modular , select worker based on lowest latency and having enough mpc
    #     if module_type == 0:
    #         candidate_node = max(roles, key=lambda x: x["mpc"])
    #
    #     # module_type = 1 when model is not modular, select the new master based on both the minimum of a latency
    #     # matrix (low latency to other good workers), and mpc
    #     elif module_type == 1:
    #         candidate_node = max(roles, key=lambda x: x["latency"])

    # def acknowledge(self):
    #     pass

    # def host_job(self, model: nn.Module):
    #     """
    #     Todo:
    #         - connect to master roles via SC and load in model
    #         - attempt to assign and relay model to other idle connected workers
    #         - determine relevant connections
    #     """
    #     self.optimizer = torch.optim.Adam
    #     self.training = True
    #     self.distribute_model(model)  # Add master vs worker functionality
