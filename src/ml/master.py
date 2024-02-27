from src.roles.worker import Worker


class Master(Worker):
    def __init__(self, host, port):
        super(Master, self).__init__(host, port, wallet_address="")
