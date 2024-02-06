from src.ml.model_analyzer import distribute_model
from src.ml.worker import Worker

import torch.nn as nn
import subprocess


class Master(Worker):
    def __init__(self, host, port):
        super(Master, self).__init__(host, port)
