from src.cryptography.rsa import get_rsa_pub_key
from src.roles.worker import Worker
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
    worker3 = Worker(host=ip, port=port + 2, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
                     debug=True)

    worker1.master = True  # We must omit this
    worker2.training = True
    worker3.training = True

    # Open ports and begin the run loop
    worker1.start()
    worker2.start()
    worker3.start()

    # Hard code workers connecting to the master node, ideally this will be done via smart contract or DHT
    worker1.connect_with_node(ip, port + 1)
    worker1.connect_with_node(ip, port + 2)

    dummy_input = torch.zeros((1, 1), dtype=torch.long)
    model = BertModel.from_pretrained("bert-base-uncased")
    # model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

    time.sleep(3)
    d_model = DistributedModel(model, worker1)
    time.sleep(3)
    nodes, graph = d_model.distribute_model()

    d_model(dummy_input)

    with open("distributed_graph.json", "w") as f:
        json.dump(graph, f, indent=4)

    worker1.stop()
    worker2.stop()
    worker3.stop()
