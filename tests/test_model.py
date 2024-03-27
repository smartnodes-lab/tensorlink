from src.cryptography.rsa import get_rsa_pub_key
from src.roles.worker import Worker
from src.ml.model_analyzer import estimate_memory
from src.ml.distributed import DistributedModel
from transformers import Wav2Vec2BertModel, BertModel
import torch.optim as optim
import torch
import json
import time


ip = "127.0.0.1"
port = 5026

mini_batch_size = 4
micro_batch_size = 2


if __name__ == "__main__":
    # Spawn 3 workers on their own ports + threads
    worker1 = Worker(host=ip, port=port, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
                     debug=True)
    worker2 = Worker(host=ip, port=port + 1, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
                     debug=False)
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

    dummy_input = torch.zeros((4, 16), dtype=torch.long)
    model = BertModel.from_pretrained("bert-base-uncased")
    # model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

    time.sleep(5)

    d_model = DistributedModel(model, worker1, mini_batch_size, micro_batch_size)
    params = d_model.parameters()
    print(1)
    # d_optimizer = optim.Adam(d_model.parameters(), lr=1e-5)
    # output = d_model(dummy_input)
    #
    # losses = [output[o][0].sum() for o in range(mini_batch_size//micro_batch_size)]
    # d_model.backward(losses)
    # d_optimizer.zero_grad()
    # d_optimizer.step()

    worker1.stop()
    worker2.stop()
