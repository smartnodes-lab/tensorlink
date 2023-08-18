from node import Node
import subprocess
import time


class SmartNode(Node):
    def __init__(self):
        super(SmartNode, self).__init__()

    def run(self):
        super().start()
