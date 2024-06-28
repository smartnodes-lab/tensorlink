import threading
import queue
import torch.optim as optim
import torch.nn as nn
import pickle


class ModelManager(threading.Thread):
    def __init__(self, task_queue):
        super(ModelManager, self).__init__()
        self.task_queue = task_queue
        self.modules = {}
        self.optimizers = {}
        self.parameters = {}

        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                task, args = self.task_queue.get(timeout=1)
                if task == 'load':
                    self.load_module(*args)
                elif task == 'write':
                    self.write_module(*args)
                self.task_queue.task_done()
            except queue.Empty:
                continue

    def load_module(self, module_name, module):
        # Load the module here (replace with your actual loading logic)
        self.modules[module_name] = module
        # Simulate loading optimizer

        self.optimizers[module_name] = optim.Adam(module.parameters())

    def write_module(self, module_name):
        # Store the module and optimizer here (replace with your actual storing logic)
        with open(f"{module_name}_module.pkl", "wb") as f:
            pickle.dump(self.modules[module_name], f)
        with open(f"{module_name}_optimizer.pkl", "wb") as f:
            pickle.dump(self.optimizers[module_name], f)
