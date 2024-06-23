from src.roles.validator import Validator
from src.roles.worker import Worker
from src.roles.user import User

from transformers import BertModel
import time


if __name__ == "__main__":

    validator = Validator(
        debug=True,
        upnp=True,
        off_chain_test=True,
        private_key="1c6768059a3e77d68a2dc3a075c93161803dbe2ad3b72069b6801a1db3a8a8f8",
    )

    user = User(
        debug=True,
        upnp=True,
        off_chain_test=True,
    )

    # worker = Worker(
    #     ip,
    #     port + 2,
    #     debug=True,
    #     upnp=False,
    #     off_chain_test=True,
    # )

    validator.start()
    # worker.start()
    user.start()

    user.connect_node(validator.rsa_key_hash, "192.168.2.237", 7779)
    time.sleep(3)
    # worker.connect_node(validator.rsa_key_hash, ip, port)

    # model = BertModel.from_pretrained("bert-base-uncased")
    # distributed_model = user.request_job(model, handle_layers=True)

    validator.stop()
    # worker.stop()
    user.stop()
