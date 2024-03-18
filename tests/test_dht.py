from src.p2p.dht_node import DHTNode
import time


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5026

    # Spawn 3 workers on their own ports + threads
    worker1 = DHTNode(host=ip, port=port, debug=True, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",)
    worker2 = DHTNode(host=ip, port=port + 1, debug=True, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",)
    worker3 = DHTNode(host=ip, port=port + 2, debug=True, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",)
    # worker4 = DHTNode(host=ip, port=port + 3, debug=True, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",)
    # worker5 = DHTNode(host=ip, port=port + 4, debug=True, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",)
    # worker6 = DHTNode(host=ip, port=port + 5, debug=True, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",)

    worker1.start()
    worker2.start()
    worker3.start()
    # worker4.start()
    # worker5.start()
    # worker6.start()

    worker2.connect_with_node(ip, port)
    worker3.connect_with_node(ip, port)
    # worker4.connect_with_node(ip, port)
    # worker5.connect_with_node(ip, port)
    # worker6.connect_with_node(ip, port)

    time.sleep(15)

    print([(con.host, con.port) for con in worker1.connections])
    print([(con.host, con.port) for con in worker2.connections])
    print([(con.host, con.port) for con in worker3.connections])

    worker1.stop()
    worker2.stop()
    worker3.stop()
