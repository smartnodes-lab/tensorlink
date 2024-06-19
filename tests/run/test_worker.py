from src.roles.worker import Worker
from transformers import BertModel


if __name__ == "__main__":
    ip = "127.0.0.1"

    worker = Worker(
        ip,
        5028,
        debug=True,
        upnp=False,
        off_chain_test=True,
    )

    worker.start()
    worker.activate()

    worker.connect_node(
        b"21c99fa3c263570d20132c24ef1b347e1b8afcdcfe88c303fb1f45b84b387a5b",
        "127.0.0.1",
        5026,
    )
