from src.roles.worker import Worker
from src.roles.validator import Validator
from transformers import BertModel


if __name__ == "__main__":

    worker = Worker(
        debug=True,
        upnp=False,
        off_chain_test=True,
    )

    validator = Validator(
        debug=True,
        upnp=False,
        off_chain_test=True,
        private_key="1c6768059a3e77d68a2dc3a075c93161803dbe2ad3b72069b6801a1db3a8a8f8",
    )

    # validator2 = Validator(
    #     ip,
    #     port + 1,
    #     debug=True,
    #     upnp=False,
    #     off_chain_test=False,
    #     private_key="d23f58a739fe6dd1390191a67e11d3d681e048168d985c9acb5609bff1f799ea",
    # )

    validator.start()
    worker.start()
    worker.activate()

    worker.connect_node(
        b"21c99fa3c263570d20132c24ef1b347e1b8afcdcfe88c303fb1f45b84b387a5b",
        "192.168.2.237",
        validator.port,
    )
