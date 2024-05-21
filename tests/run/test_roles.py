from src.roles.user import User
from src.roles.worker import Worker
from src.roles.validator import Validator


if __name__ == "__main__":
    local_host = "127.0.0.1"

    user = User(
        host=local_host,
        port=5025,
        private_key="ead8bd63b4e3e2e3aa846b1f0dfa602d0c61ef79e0b32c180cbc5ceb0828f1f6",
        debug=True,
        upnp=False,
    )

    validator1 = Validator(
        host=local_host,
        port=5026,
        private_key="afabc26f69943849b267cd38836c0cab241998a59e1e81010d130b899e2e08d2",
        debug=True,
        upnp=False,
    )

    validator2 = Validator(
        host=local_host,
        port=5027,
        private_key="af9834c2cf887fbdf236d03691de043c17353eb7cafff451bb3a7bd949c6f0be",
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
    validator2.start()
    worker1.start()

    # Initial seed connection
    user.connect_dht_node(local_host, 5026)  # Simulate a seed node connection
    validator2.connect_dht_node(local_host, 5026)
    worker1.connect_dht_node(local_host, 5026)

    # Bootstrap
    validator1.bootstrap()
    validator2.bootstrap()
    worker1.bootstrap()
    user.bootstrap()

    user.stop()
    validator1.stop()
    validator2.stop()
    worker1.stop()
