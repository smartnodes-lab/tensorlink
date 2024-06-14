from src.roles.worker import Worker
from transformers import BertModel


if __name__ == "__main__":
    ip = "127.0.0.1"

    worker = Worker(
        ip,
        5028,
        debug=True,
        upnp=False,
        off_chain_test=False,
    )

    worker.start()
    worker.activate()

    val_id = worker.contract.functions.validatorKeyById(1).call()
    worker.connect_node(val_id, "127.0.0.1", 5026)
