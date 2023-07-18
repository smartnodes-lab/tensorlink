from src.node import Node

import numpy as np
import random


def test_node_cascade():
    batch_size = 32
    input_size = [4, 256]

    # Job author
    master_node = Node(id="master-node", host="localhost", port=9091, debug=True)

    # Connected workers
    worker_node1 = Node(id="worker-node-1", host="localhost", port=9092, debug=False)
    worker_node2 = Node(id="worker-node-2", host="localhost", port=9093, debug=False)

    # Start nodes
    master_node.start()
    worker_node1.start()
    worker_node2.start()

    # Worker nodes grab master node info from blockchain and connect
    worker_node1.connect_with_node(master_node.host, master_node.port, reconnect=True)
    worker_node2.connect_with_node(master_node.host, master_node.port, reconnect=True)

    # Randomize the order of workers for workflow
    random.shuffle(master_node.nodes_inbound)

    # Attempt 3 training epochs
    for i in range(3):
        # Master node forward pass placeholder
        x = np.array(
            [np.array([1 for _ in range(input_size[0])])
             for _ in range(input_size[1]) for _ in range(batch_size)]
        ).tobytes()

        for n in range(len(master_node.nodes_inbound)):
            master_node.send_to_node(master_node.nodes_inbound[n], x)


test_node_cascade()
