from src.roles.user import User
from src.roles.worker import Worker
from src.roles.validator import Validator


if __name__ == "__main__":
    local_host = "127.0.0.1"

    user = User(
        host=local_host,
        port=5025,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
        debug=True,
    )

    validator1 = Validator(
        host=local_host,
        port=5026,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
        debug=True,
    )

    validator2 = Validator(
        host=local_host,
        port=5027,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
        debug=True,
    )

    worker1 = Worker(
        host=local_host,
        port=5028,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    )

    user.start()
    validator1.start()
    validator2.start()
    worker1.start()

    user.connect_dht_node(local_host, 5026)  # Simulate a seed node connection
    validator2.connect_dht_node(local_host, 5026)

    validator1.bootstrap()
    validator2.bootstrap()
    user.bootstrap()

    user.stop()
    validator1.stop()
    validator2.stop()
    worker1.stop()
