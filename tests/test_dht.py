import hashlib

from src.roles.worker import Worker
import time


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5026

    # Spawn 3 workers on their own ports + threads
    worker1 = Worker(
        host=ip,
        port=port,
        debug=True,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    )

    worker2 = Worker(
        host=ip,
        port=port + 1,
        debug=False,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    )
    worker3 = Worker(
        host=ip,
        port=port + 2,
        debug=False,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    )

    worker1.start()
    worker2.start()
    worker3.start()

    worker2.connect_dht_node(ip, port)
    worker3.connect_dht_node(ip, port)

    time.sleep(12)
    key = hashlib.sha256(b"a").hexdigest().encode()
    worker2.store_key_value_pair(key, "TEST VALUE")

    print(worker1.query_routing_table(key))
    print([(con.host, con.port) for con in worker1.connections])
    print([(con.host, con.port) for con in worker2.connections])
    # print([(con.host, con.port) for con in worker3.connections])

    worker1.stop()
    worker2.stop()
    worker3.stop()
