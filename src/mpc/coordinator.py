from src.roles.user import User
from src.roles.validator import Validator
from src.roles.worker import Worker
from src.ml.distributed import DistributedModel
from src.ml.worker import DistributedWorker

import multiprocessing
import threading
import atexit


class BaseCoordinator:
    _instance = None

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super(BaseCoordinator, cls).__new__(cls)
    #         cls._instance._initialized = False
    #         cls._instance._init_kwargs = kwargs
    #     return cls._instance

    def __init__(self, **kwargs):
        # if self._initialized:
        #     return

        self.node_requests = multiprocessing.Queue()
        self.node_responses = multiprocessing.Queue()

        self.init_kwargs = kwargs

        self.node_process = None

        self._initialized = True
        self.setup()
        atexit.register(self.cleanup)

    def setup(self):
        multiprocessing.set_start_method("spawn", force=True)
        self.node_process = multiprocessing.Process(target=self.run_role)
        self.node_process.start()

    def cleanup(self):
        if self.node_process is not None:
            self.node_process.terminate()
            self.node_process.join()

    def send_request(self, request_type, args):
        """
        Sends a request to the node and waits for the response.
        """
        request = {"type": request_type, "args": args}
        try:
            self.node_requests.put(request)
            response = self.node_responses.get()  # Blocking call, waits for response
            return response["return"]
        except Exception as e:
            print(f"Error sending request: {e}")
            return {"error": str(e)}

    def run_role(self):
        raise NotImplementedError("Subclasses must implement this method")


class DistributedCoordinator(BaseCoordinator):
    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update({
            'debug': kwargs.get('debug', True),
            'upnp': kwargs.get('upnp', False),
            'off_chain_test': kwargs.get('off_chain_test', True)
        })
        role_instance = User(
            self.node_requests,
            self.node_responses,
            **kwargs
        )
        role_instance.start()
        self.node_process = role_instance

    def create_distributed_model(self, model, n_pipelines, max_module_size=4e9, config=None):
        dist_model = DistributedModel(self.node_requests, self.node_responses, model, n_pipelines)

        distribution = dist_model.parse_model(model, max_module_size)
        distributed_config = self.send_request("request_job", (n_pipelines, distribution))
        dist_model.distribute_model(distributed_config)
        return dist_model


class ValidatorCoordinator(BaseCoordinator):
    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update({
            'debug': kwargs.get('debug', True),
            'upnp': kwargs.get('upnp', False),
            'off_chain_test': kwargs.get('off_chain_test', True)
        })
        role_instance = Validator(
            self.node_requests,
            self.node_responses,
            **kwargs
        )
        role_instance.start()
        self.node_process = role_instance


class WorkerCoordinator(BaseCoordinator):
    def run_role(self):
        kwargs = self.init_kwargs.copy()
        kwargs.update({
            'debug': kwargs.get('debug', True),
            'upnp': kwargs.get('upnp', False),
            'off_chain_test': kwargs.get('off_chain_test', True)
        })
        role_instance = Worker(
            self.node_requests,
            self.node_responses,
            **kwargs
        )
        role_instance.start()
        role_instance.activate()
        self.node_process = role_instance

        distributed_worker = DistributedWorker(self.node_requests, self.node_responses)
        distributed_worker.run()
        self.distributed_worker = distributed_worker
