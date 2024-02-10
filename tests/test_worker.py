from src.ml.worker import Worker


ip = "127.0.0.1"
port = 5026


master = Worker(host=ip, port=port, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh", debug=True)
# worker1 = Worker(host=ip, port=port + 1, debug=True)

master.start()
# worker1.start()

# master.connect_with_node(ip, port + 1)
# worker1.connect_with_node(ip, port)

master.stop()
# worker1.stop()
