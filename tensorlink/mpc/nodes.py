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
    for cursor in itertools.cycle('|/-\\'):
        yield cursor


def show_spinner(stop_event, message="Processing"):
    """Displays a spinner in the console."""
    spinner = spinning_cursor()
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r")  # Clear the spinner
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

        signal.signal(signal.SIGINT, self.signal_handler)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup)

        self._initialized = True
        self.setup()

    def setup(self):
        self.node_process = multiprocessing.Process(target=self.run_role, daemon=True)
        self.node_process.start()

    def signal_handler(self, sig, frame):
        """Handle termination signals and call cleanup, most importantly removing open port mappings"""
        print(f"Received signal {sig}. Cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        # Process cleanup
        if self.node_process is not None and self.node_process.exitcode is None:
            # Send a stop request to the role instance
            self.send_request("stop", (None,), timeout=3)
            self.node_process.join(timeout=10)

            # If the process is still alive, terminate it
            if self.node_process.is_alive():
                print("Forcing termination for node process.")
                self.node_process.terminate()

            # Final join to ensure it's completely shut down
            self.node_process.join()
            self.node_process = None  # Reset to None after cleanup

        # UPnP port mapping cleanup
        if self.upnp_enabled:
            try:
                upnp = miniupnpc.UPnP()
                upnp.discoverdelay = 200

                # Discover UPnP devices
                devices_found = upnp.discover()
                if devices_found == 0:
                    # print("No UPnP devices found.")
                    return

                upnp.selectigd()  # Select Internet Gateway Device
                local_ip = upnp.lanaddr
                removed_count = 0

                # print("Scanning existing UPnP port mappings...")
                i = 0
                while True:
                    try:
                        mapping = upnp.getgenericportmapping(i)
                        if mapping is None:
                            break  # No more mappings

                        ext_port, protocol, (int_ip, int_port), desc = mapping[:4]

                        # Check if mapping matches our application
                        if int_ip == local_ip and desc == "SmartNode":
                            # print(f"Removing UPnP mapping: {ext_port}/{protocol} -> {int_ip}:{int_port}")
                            upnp.deleteportmapping(ext_port, protocol)
                            removed_count += 1

                        i += 1

                    except Exception:
                        break  # End of list or error during retrieval

                # print(f"Cleanup complete. Removed {removed_count} UPnP mappings.")

            except Exception as e:
                print(f"Error during UPnP cleanup: {e}")

    def send_request(self, request_type, args, timeout=None):
        """
        Sends a request to the roles and waits for the response.
        """
        request = {"type": request_type, "args": args}
        response = None
        try:
            self.mpc_lock.acquire(timeout=timeout)
            self.node_requests.put(request)
            response = self.node_responses.get()  # Blocking call, waits for response

        except Exception as e:
            print(f"Error sending request: {e}")
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
        role_instance = Worker(
            self.node_requests,
            self.node_responses,
            **kwargs
        )
        role_instance.start()
        role_instance.activate()
        role_instance.join()

    def setup(self):
        super().setup()
        distributed_worker = DistributedWorker(self.node_requests, self.node_responses, self.mpc_lock)
        t = threading.Thread(target=distributed_worker.run)
        t.start()


class ValidatorNode(BaseNode):
    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update({
            'upnp': kwargs.get('upnp', True),
            'off_chain_test': kwargs.get('off_chain_test', False)
        })
        role_instance = Validator(
            self.node_requests,
            self.node_responses,
            **kwargs
        )
        role_instance.start()
        role_instance.join()


class UserNode(BaseNode):
    def run_role(self):
        kwargs = self.init_kwargs.copy()

        role_instance = User(
            self.node_requests,
            self.node_responses,
            **kwargs
        )
        role_instance.start()
        role_instance.join()

    def create_distributed_model(self, model, n_pipelines, optimizer_type=None, dp_factor=None):
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=show_spinner, args=(stop_spinner, "Creating distributed model"))

        try:
            # Start the spinner
            spinner_thread.start()

            dist_model = DistributedModel(self.node_requests, self.node_responses, self.mpc_lock, model, n_pipelines)
            self.send_request("request_workers", None)
            time.sleep(3)
            workers = self.send_request("check_workers", None)
            if len(workers) == 0:
                self.send_request("request_workers", None)
                time.sleep(5)
                workers = self.send_request("check_workers", None)

            if optimizer_type is None:
                optimizer_type = torch.optim.Adam

            dist_model.worker_info = workers
            if len(workers) == 0:
                self.send_request("debug_print", (
                    "Job creation failed: network at capacity (not enough workers)!", "bright_red", logging.CRITICAL))
                return None, None

            distribution = dist_model.parse_model(model, handle_layer=False)
            distributed_config = self.send_request("request_job", (n_pipelines, 1, distribution))

            if distributed_config:
                for module_id, module_info in distributed_config.items():
                    if module_info["type"] == "offloaded":
                        module, module_name = access_module(model, module_info["mod_id"])
                        setattr(module, "optimizer", optimizer_type)
            else:
                print("Could not obtain job from network... Please try again.")
                return False

            dist_model.distribute_model(distributed_config)

            def _create_distributed_optimizer(**optimizer_kwargs):
                return create_distributed_optimizer(dist_model, optimizer_type, **optimizer_kwargs)

            setattr(self, "distributed_model", dist_model)

            return dist_model, _create_distributed_optimizer

        finally:
            # Stop the spinner
            stop_spinner.set()
            spinner_thread.join()

    def cleanup(self):
        """Downloads parameters from workers before shutting down"""
        if hasattr(self, "distributed_model"):
            if self.distributed_model.training:
                self.distributed_model.parameters(distributed=True, load=False)

        super().cleanup()
