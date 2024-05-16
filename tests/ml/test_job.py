from src.roles.user import User
from src.roles.worker import Worker
from src.roles.validator import Validator
from transformers import BertModel


if __name__ == "__main__":
    local_host = "127.0.0.1"

    user = User(
        host=local_host,
        port=5025,
        private_key="f0cc12573626885435c1f1c09b067b23c892a49e0f976ffa5eded0c0021e8658",
        debug=True,
        upnp=False,
    )

    validator1 = Validator(
        host=local_host,
        port=5026,
        private_key="a21213a1e5d5531b990b53159018f63336d8ffcac272ab79a80e7a6ab9e7f242",
        debug=True,
        upnp=False,
    )

    worker1 = Worker(
        host=local_host,
        port=5029,
        private_key="ec79739589c5d236a2d04d8884dd93f4493efe95b1844b96013307ac6cdd101a",
        debug=True,
        upnp=False,
    )

    # Start up test nodes
    user.start()
    validator1.start()
    worker1.start()

    worker1.training = True

    # Initial seed connection
    user.connect_dht_node(local_host, 5026)  # Simulate a seed node connection
    worker1.connect_dht_node(local_host, 5026)

    # Bootstrap
    worker1.bootstrap()
    user.bootstrap()

    # User requests job
    model = BertModel.from_pretrained("bert-base-uncased")
    d_model = user.request_job(model, 1, 1)

    user.stop()
    validator1.stop()
    worker1.stop()
