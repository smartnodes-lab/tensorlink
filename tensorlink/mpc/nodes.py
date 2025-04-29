import logging
import signal
import sys
import threading
import time
import torch.multiprocessing as mp

from tensorlink.ml.worker import DistributedWorker
from tensorlink.ml.validator import DistributedValidator
from tensorlink.nodes.user import User
from tensorlink.nodes.validator import Validator
from tensorlink.nodes.worker import Worker


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


mp.set_start_method("spawn", force=True)


class BaseNode:
    def __init__(
        self,
        upnp=True,
        max_connections: int = 0,
        off_chain_test=False,
        local_test=False,
        print_level=logging.WARNING,
        trusted=False,
        utilization=True,
    ):
        self.node_requests = mp.Queue()
        self.node_responses = mp.Queue()
        self.mpc_lock = mp.Lock()

        self.init_kwargs = {
            "print_level": print_level,
            "max_connections": max_connections,
            "upnp": upnp,
            "off_chain_test": off_chain_test,
            "local_test": local_test,
        }
        self.trusted = trusted
        self.upnp_enabled = upnp
        self.utilization = utilization

        self.node_process = None
        self.node_instance = None

        self._stop_event = mp.Event()
        self._setup_signal_handlers()
        self._initialized = True
        self.start()

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

    def start(self):
        self.node_process = mp.Process(target=self.run_role, daemon=True)
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

    def send_request(self, request_type, args, timeout=5):
        """
        Sends a request to the roles and waits for the response.
        """
        request = {"type": request_type, "args": args}

        try:
            self.mpc_lock.acquire(timeout=timeout)
            self.node_requests.put(request)
            response = self.node_responses.get(
                timeout=timeout
            )  # Blocking call, waits for response

        except Exception as e:
            print(f"Error sending '{request_type}' request: {e}")
            response = {"return": str(e)}

        finally:
            self.mpc_lock.release()

        return response["return"]

    def run_role(self):
        raise NotImplementedError("Subclasses must implement this method")

    def connect_node(self, host: str, port: int, node_id: str = None, timeout: int = 5):
        if node_id is None:
            node_id = ""

        self.send_request("connect_node", (node_id, host, port), timeout=timeout)


class WorkerNode(BaseNode):
    distributed_worker = None

    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update(
            {
                "upnp": kwargs.get("upnp", True),
                "off_chain_test": kwargs.get("off_chain_test", False),
            }
        )

        node_instance = Worker(self.node_requests, self.node_responses, **kwargs)
        try:
            node_instance.activate()
            node_instance.run()

            while node_instance.is_alive():
                time.sleep(1)

        except KeyboardInterrupt:
            node_instance.stop()

    def start(self):
        super().start()
        distributed_worker = DistributedWorker(self, trusted=self.trusted)
        if self.utilization:
            t = threading.Thread(target=distributed_worker.run, daemon=True)
            t.start()
            time.sleep(3)
        else:
            distributed_worker.run()


class ValidatorNode(BaseNode):
    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update(
            {
                "upnp": kwargs.get("upnp", True),
                "off_chain_test": kwargs.get("off_chain_test", False),
            }
        )

        node_instance = Validator(self.node_requests, self.node_responses, **kwargs)

        try:
            node_instance.run()

            while node_instance.is_alive():
                time.sleep(1)

        except KeyboardInterrupt:
            node_instance.stop()

    def start(self):
        super().start()
        distributed_validator = DistributedValidator(self, trusted=self.trusted)
        if self.utilization:
            t = threading.Thread(target=distributed_validator.run, daemon=True)
            t.start()
            time.sleep(3)
        else:
            distributed_validator.run()


class UserNode(BaseNode):
    def run_role(self):
        kwargs = self.init_kwargs.copy()

        node_instance = User(self.node_requests, self.node_responses, **kwargs)
        try:
            node_instance.run()

            while node_instance.is_alive():
                time.sleep(1)

        except KeyboardInterrupt:
            node_instance.stop()

    def cleanup(self):
        """Downloads parameters from workers before shutting down"""
        if hasattr(self, "distributed_model"):
            if self.distributed_model.training:
                self.distributed_model.parameters(distributed=True, load=False)

        super().cleanup()
