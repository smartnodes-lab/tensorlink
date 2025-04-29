from tensorlink import ValidatorNode, WorkerNode
import requests
import time


if __name__ == "__main__":
    """Test the /api/request-job endpoint"""

    # Start by setting up the network simulation
    worker = WorkerNode(local_test=True)
    validator = ValidatorNode(local_test=True)

    # Get validator credentials for direct connection
    val_key, val_host, val_port = validator.send_request("info", None)
    worker.connect_node(val_host, val_port, node_id=val_key, timeout=5)

    # Ensures validator has enough time to start up and load some models
    time.sleep(10)
