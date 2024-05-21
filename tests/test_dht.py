from src.roles.validator import Validator
from src.roles.user import User
import hashlib
import time


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5026

    # Spawn 3 workers on their own ports + threads
    node1 = Validator(
        host=ip,
        port=port,
        debug=True,
        private_key="fbb703ce65552e6ece9e0fa3c9cd25586dea8642e48dc902c163019b786f835e",
        upnp=False,
    )

    node2 = Validator(
        host=ip,
        port=port + 1,
        debug=True,
        private_key="58644704af6595102792f52ea528cc3970c2fb98e9301773567072e548a285bb",
        upnp=False,
    )

    node3 = Validator(
        host=ip,
        port=port + 2,
        debug=False,
        private_key="495e68a235edd6cec43b9f129c01dede5d5bb67d1e1f208cec5cc970a5bfda07",
        upnp=False,
    )

    user = User(
        host=ip,
        port=5025,
        private_key="f0cc12573626885435c1f1c09b067b23c892a49e0f976ffa5eded0c0021e8658",
        debug=True,
        upnp=False,
    )

    node1.start()
    node2.start()
    node3.start()
    user.start()

    node2.connect_dht_node(ip, port)
    node3.connect_dht_node(ip, port)
    user.connect_dht_node(ip, port)

    node2.bootstrap()
    node3.bootstrap()
    user.bootstrap()

    print(node1.connections)
    print(node2.connections)
    print(node3.connections)

    node1.request_worker_stats()

    node1.stop()
    node2.stop()
    node3.stop()
    user.stop()
