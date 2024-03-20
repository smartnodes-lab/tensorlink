from src.p2p.smart_node import SmartDHTNode
import time


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5026

    # Spawn 3 workers on their own ports + threads
    worker1 = SmartDHTNode(host=ip, port=port, debug=True, public_key="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh")
    worker2 = SmartDHTNode(host=ip, port=port + 1, debug=True, public_key="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh")
    worker3 = SmartDHTNode(host=ip, port=port + 2, debug=False, public_key="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh")

    worker1.start()
    worker2.start()
    worker3.start()

    worker2.connect_dht_node(ip, port)
    worker3.connect_dht_node(ip, port)

    time.sleep(15)

    print([(con.host, con.port) for con in worker1.connections])
    print([(con.host, con.port) for con in worker2.connections])
    print([(con.host, con.port) for con in worker3.connections])

    worker1.stop()
    worker2.stop()
    worker3.stop()
