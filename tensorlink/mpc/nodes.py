from tensorlink.roles.user import User
from tensorlink.roles.validator import Validator
from tensorlink.roles.worker import Worker
from tensorlink.ml.module import DistributedModel
from tensorlink.ml.worker import DistributedWorker
from tensorlink.ml.optim import create_distributed_optimizer
from tensorlink.ml.graphing import handle_layers
from tensorlink.ml.utils import access_module

import multiprocessing
import torch
import time

# Set the start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)


class BaseNode:
    _instance = None

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super(BaseNode, cls).__new__(cls)
    #         cls._instance._initialized = False
    #         cls._instance._init_kwargs = kwargs
    #     return cls._instance

    def __init__(self, **kwargs):
        # if self._initialized:
        #     return

        self.node_requests = multiprocessing.Queue()
        self.node_responses = multiprocessing.Queue()
        self.mpc_lock = multiprocessing.Lock()

        self.init_kwargs = kwargs

        self.node_process = None

        self._initialized = True
        self.setup()

    def setup(self):
        self.node_process = multiprocessing.Process(target=self.run_role, daemon=True)
        self.node_process.start()

    def cleanup(self):
        if self.node_process is not None:
            # Send a stop request to the role instance
            self.send_request("stop", (None,))
            self.node_process.join(timeout=10)

            # If the process is still alive, terminate it
            if self.node_process.is_alive():
                print("Forcing termination for node process.")
                self.node_process.terminate()

            # Final join to ensure it's completely shut down
            self.node_process.join()
            self.node_process = None  # Reset to None after cleanup

    def send_request(self, request_type, args):
        """
        Sends a request to the roles and waits for the response.
        """
        request = {"type": request_type, "args": args}
        try:
            self.mpc_lock.acquire()
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


class ValidatorNode(BaseNode):
    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update({
            'debug': kwargs.get('debug', True),
            'upnp': kwargs.get('upnp', True),
            'off_chain_test': kwargs.get('off_chain_test', False)
        })
        role_instance = Validator(
            self.node_requests,
            self.node_responses,
            **kwargs
        )
        role_instance.start()
        self.node_process = role_instance


class UserNode(BaseNode):
    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update({
            'debug': kwargs.get('debug', False),
            'upnp': kwargs.get('upnp', True),
            'off_chain_test': kwargs.get('off_chain_test', False)
        })
        role_instance = User(
            self.node_requests,
            self.node_responses,
            **kwargs
        )
        role_instance.start()
        self.node_process = role_instance

    def create_distributed_model(self, model, n_pipelines, optimizer_type=None, dp_factor=None):  # TODO Max module size etc?
        dist_model = DistributedModel(self.node_requests, self.node_responses, self.mpc_lock, model, n_pipelines)
        self.send_request("request_workers", None)
        time.sleep(3)
        workers = self.send_request("check_workers", None)
        if len(workers) == 0:
            time.sleep(5)
            workers = self.send_request("check_workers", None)

        if optimizer_type is None:
            optimizer_type = torch.optim.Adam

        dist_model.worker_info = workers
        distribution = dist_model.parse_model(model, handle_layer=False)
        distributed_config = self.send_request("request_job", (n_pipelines, 1, distribution))

        for module_id, module_info in distributed_config.items():
            if module_info["type"] == "offloaded":
                module, module_name = access_module(model, module_info["mod_id"])
                setattr(module, "optimizer", optimizer_type)

        dist_model.distribute_model(distributed_config)

        def _create_distributed_optimizer(**optimizer_kwargs):
            return create_distributed_optimizer(dist_model, optimizer_type, **optimizer_kwargs)

        return dist_model, _create_distributed_optimizer


class WorkerNode(BaseNode):
    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update({
            'debug': kwargs.get('debug', False),
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
        self.node_process = role_instance

        self.run_distributed_worker()

    def run_distributed_worker(self):
        distributed_worker = DistributedWorker(self.node_requests, self.node_responses, self.mpc_lock)
        distributed_worker = multiprocessing.Process(target=distributed_worker.run)
        distributed_worker.run()
