from src.roles.validator import Validator
from src.roles.worker import Worker
from src.roles.user import User

from transformers import BertModel
import time


if __name__ == "__main__":

    # validator = Validator(
    #     debug=True,
    #     upnp=False,
    #     off_chain_test=True,
    #     private_key="1c6768059a3e77d68a2dc3a075c93161803dbe2ad3b72069b6801a1db3a8a8f8",
    # )

    user = User(
        debug=True,
        upnp=False,
        off_chain_test=True,
    )

    #9 worker = Worker(
    #     debug=True,
    #     upnp=False,
    #     off_chain_test=True,
    # )

    # validator.start()
    # worker.start()
    user.start()

    # worker.activate()

    user.connect_node(b"asd", "192.168.2.237", 38752)
    # worker.connect_node(validator.rsa_key_hash, validator.host, validator.port)

    model = BertModel.from_pretrained("bert-base-uncased")
    # user.send_module(list(model.children())[1], user.nodes[user.workers[0]])
    distributed_model = user.request_job(model, handle_layers=True)

    # validator.stop()
    # worker.stop()
    # user.stop()
