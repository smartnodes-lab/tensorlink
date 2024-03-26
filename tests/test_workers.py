from src.cryptography.rsa import get_rsa_pub_key
from src.roles.worker import Worker
from src.ml.model_analyzer import estimate_memory
from src.ml.distributed import DistributedModel
from transformers import Wav2Vec2BertModel, BertModel
import torch
import json
import time


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5026

    # Spawn 3 workers on their own ports + threads
    worker1 = Worker(host=ip, port=port, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
                     debug=True)
    worker2 = Worker(host=ip, port=port + 1, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
                     debug=True)
    # worker3 = Worker(host=ip, port=port + 2, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
    #                  debug=True)

    worker1.master = True  # We must omit this
    worker2.training = True
    # worker3.training = True

    # Open ports and begin the run loop
    worker1.start()
    worker2.start()
    # worker3.start()

    # Hard code workers connecting to the master node, ideally this will be done via smart contract or DHT
    worker1.connect_dht_node(ip, port + 1)
    # worker1.connect_dht_node(ip, port + 2)

    worker1.stop()
    worker2.stop()
