import multiprocessing
import signal
import sys
from  import node_process
from .model import DistributedModel
from .model_manager import model_manager_process


class ProcessManager:
    def __init__(self, model):
        self.command_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.node_process = multiprocessing.Process(target=node_process, args=(self.command_queue,))
        self.model_manager_process = multiprocessing.Process(target=model_manager_process,
                                                             args=(self.command_queue, self.response_queue))
        self.model = DistributedModel(model)

        signal.signal(signal.SIGINT, self.terminate_processes)
        signal.signal(signal.SIGTERM, self.terminate_processes)

    def start_processes(self):
        self.node_process.start()
        print(f"Started node process with PID {self.node_process.pid}")

        self.model_manager_process.start()
        print(f"Started model manager process with PID {self.model_manager_process.pid}")

        self.model.start()
        print(f"Started model process with PID {self.model.process.pid}")

        self.node_process.join()
        self.model_manager_process.join()
        self.model.join()

    def terminate_processes(self, sig, frame):
        print('Terminating processes...')
        self.command_queue.put(('TERMINATE', None))
        self.node_process.terminate()
        self.model_manager_process.terminate()
        self.model.process.terminate()
        self.node_process.join()
        self.model_manager_process.join()
        self.model.process.join()
        sys.exit(0)

    def store_model(self, model_id, model):
        self.command_queue.put(('STORE_MODEL', (model_id, model)))
        response = self.response_queue.get()
        return response


def start(model):
    manager = ProcessManager(model)
    manager.start_processes()
    return manager
