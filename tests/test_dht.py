from src.p2p.smart_node import SmartDHTNode
import hashlib
import time


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5026

    # Spawn 3 workers on their own ports + threads
    node1 = SmartDHTNode(
        host=ip,
        port=port,
        debug=True,
        public_key="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    )

    node2 = SmartDHTNode(
        host=ip,
        port=port + 1,
        debug=False,
        public_key="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    )

    node3 = SmartDHTNode(
        host=ip,
        port=port + 2,
        debug=False,
        public_key="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    )

    node1.start()
    node2.start()
    node3.start()

    node2.connect_dht_node(ip, port)
    node3.connect_dht_node(ip, port)

    node2.bootstrap()

    time.sleep(12)
    key = hashlib.sha256(b"a").hexdigest().encode()
    node2.store_key_value_pair(key, "TEST VALUE")

    print(node1.query_routing_table(key))
    print([(con.host, con.port) for con in node1.connections])
    print([(con.host, con.port) for con in node2.connections])
    # print([(con.host, con.port) for con in worker3.connections])

    node1.stop()
    node2.stop()
    node3.stop()
