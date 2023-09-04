from p2p.node import Node
import time


ip = "127.0.0.1"
port = 5026

node = Node(
    host=ip,
    port=port,
    debug=True
)

node2 = Node(
    host=ip,
    port=port + 1,
    debug=True
)

node.start()
node2.start()

node.connect_with_node(ip, port + 1)

time.sleep(0.1)

start_time = str(time.time()).encode()
node.send_to_nodes(start_time)

node.stop()
node2.stop()
