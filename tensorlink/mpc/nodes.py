from tensorlink.roles.user import User
from tensorlink.roles.validator import Validator
from tensorlink.roles.worker import Worker
from tensorlink.ml.module import DistributedModel
from tensorlink.ml.worker import DistributedWorker
from tensorlink.ml.optim import create_distributed_optimizer
from tensorlink.ml.graphing import handle_layers
from tensorlink.ml.utils import access_module

import multiprocessing
import threading
import miniupnpc
import itertools
import logging
import signal
import atexit
import torch
import time
import sys


def spinning_cursor():
    """Generator for a spinning cursor animation."""
    for cursor in "|/-\\":
        yield cursor


def show_spinner(stop_event, message="Processing"):
    """
    Displays a spinner in the console.

    Args:
        stop_event (threading.Event): Event to signal when to stop the spinner.
        message (str): The message to display alongside the spinner.
    """
    spinner = spinning_cursor()
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")  # Clear the line
    sys.stdout.flush()


# Set the start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)


class BaseNode:
    def __init__(
            self,
            upnp=True,
            max_connections: int = 0,
            off_chain_test=False,
            local_test=False,
            print_level=logging.WARNING
    ):
        self.node_requests = multiprocessing.Queue()
        self.node_responses = multiprocessing.Queue()
        self.mpc_lock = multiprocessing.Lock()

        self.init_kwargs = {
            "print_level": print_level,
            "max_connections": max_connections,
            "upnp": upnp,
            "off_chain_test": off_chain_test,
            "local_test": local_test
        }

        self.upnp_enabled = upnp

        self.node_process = None
        self.node_instance = None

        self._stop_event = multiprocessing.Event()
        self._setup_signal_handlers()
        self._initialized = True
        self.setup()

    def _setup_signal_handlers(self):
        """
        Set up signal handlers for graceful shutdown.
        Uses a multiprocessing Event to signal across processes.
        """
        def handler(signum, frame):
            print(f"Received signal {signum}. Initiating shutdown...")
            self._stop_event.set()
            self.cleanup()
            sys.exit(0)

        # Register handlers for common termination signals
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
            signal.signal(sig, handler)

    def setup(self):
        self.node_process = multiprocessing.Process(target=self.run_role, daemon=True)
        self.node_process.start()

    def cleanup(self):
        # Process cleanup
        if self.node_process is not None and self.node_process.exitcode is None:
            # Send a stop request to the role instance
            response = self.send_request("stop", (None,), timeout=15)
            if response:
                self.node_process.join(timeout=15)

            # If the process is still alive, terminate it
            if self.node_process.is_alive():
                print("Forcing termination for node process.")
                self.node_process.terminate()

            # Final join to ensure it's completely shut down
            self.node_process.join()
            self.node_process = None  # Reset to None after cleanup

    def send_request(self, request_type, args, timeout=3):
        """
        Sends a request to the roles and waits for the response.
        """
        request = {"type": request_type, "args": args}
        response = None
        try:
            self.mpc_lock.acquire(timeout=timeout)
            self.node_requests.put(request)
            response = self.node_responses.get(timeout=timeout)  # Blocking call, waits for response

        except Exception as e:
            print(f"Error sending '{request_type}' request: {e}")
            response = {"return": str(e)}

        finally:
            self.mpc_lock.release()

        return response["return"]

    def run_role(self):
        raise NotImplementedError("Subclasses must implement this method")


class WorkerNode(BaseNode):
    distributed_worker = None

    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update({
            'upnp': kwargs.get('upnp', True),
            'off_chain_test': kwargs.get('off_chain_test', False)
        })

        node_instance = Worker(
            self.node_requests,
            self.node_responses,
            **kwargs
        )
        try:
            node_instance.activate()
            node_instance.run()

            while node_instance.is_alive():
                time.sleep(1)

        except KeyboardInterrupt:
            node_instance.stop()

    def setup(self):
        super().setup()
        distributed_worker = DistributedWorker(self.node_requests, self.node_responses, self.mpc_lock)
        t = threading.Thread(target=distributed_worker.run, daemon=True)
        t.start()


class ValidatorNode(BaseNode):
    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update({
            'upnp': kwargs.get('upnp', True),
            'off_chain_test': kwargs.get('off_chain_test', False)
        })
        node_instance = Validator(
            self.node_requests,
            self.node_responses,
            **kwargs
        )

        try:
            node_instance.run()

            while node_instance.is_alive():
                time.sleep(1)

        except KeyboardInterrupt:
            node_instance.stop()


class UserNode(BaseNode):
    def run_role(self):
        kwargs = self.init_kwargs.copy()

        node_instance = User(
            self.node_requests,
            self.node_responses,
            **kwargs
        )
        try:
            node_instance.run()

            while node_instance.is_alive():
                time.sleep(1)

        except KeyboardInterrupt:
            node_instance.stop()

    def create_distributed_model(self, model, training, n_pipelines=1, optimizer_type=None, dp_factor=None):
        # stop_spinner = threading.Event()
        # spinner_thread = threading.Thread(target=show_spinner, args=(stop_spinner, "Creating distributed model"))

        try:
            # Start the spinner
            # spinner_thread.start()

            dist_model = DistributedModel(self.node_requests, self.node_responses, self.mpc_lock, model, n_pipelines)
            # self.send_request("request_workers", None)
            # time.sleep(3)
            # workers = self.send_request("check_workers", None)
            # if len(workers) == 0:
            #     self.send_request("request_workers", None)
            #     time.sleep(5)
            #     workers = self.send_request("check_workers", None)

            if optimizer_type is None:
                optimizer_type = torch.optim.Adam

            # dist_model.worker_info = workers
            # if len(workers) == 0:
            #     self.send_request("debug_print", (
            #         "Job creation failed: network at capacity (not enough workers)!", "bright_red", logging.CRITICAL))
            #     return None, None

            dist_model.training = training
            distribution = dist_model.parse_model(model, handle_layer=False)

            if training:
                for module_id, module in distribution.items():
                    if module["type"] == "offloaded":
                        module["optimizer"] = f"{optimizer_type.__module__}.{optimizer_type.__name__}"
                        module["training"] = training

            distributed_config = self.send_request("request_job", (n_pipelines, 1, distribution),
                                                   timeout=10)

            if not distributed_config:
                print("Could not obtain job from network... Please try again.")
                return False

            dist_model.distribute_model(distributed_config)

            def _create_distributed_optimizer(**optimizer_kwargs):
                return create_distributed_optimizer(dist_model, optimizer_type, **optimizer_kwargs)

            setattr(self, "distributed_model", dist_model)

            return dist_model, _create_distributed_optimizer

        finally:
            pass
            # Stop the spinner
            # stop_spinner.set()
            # spinner_thread.join()

    def cleanup(self):
        """Downloads parameters from workers before shutting down"""
        if hasattr(self, "distributed_model"):
            if self.distributed_model.training:
                self.distributed_model.parameters(distributed=True, load=False)

        super().cleanup()
