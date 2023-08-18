from p2p.node import Node
import time


ip = "127.0.0.1"
port = 5025

node = Node(ip, port, debug=True)
node2 = Node(ip, 5026, debug=True)

node.start()
node2.start()

node2.connect_with_node(ip, port)

time.sleep(0.1)

start_time = str(time.time()).encode()
node2.send_to_nodes(start_time)

node.stop()
node2.stop()
