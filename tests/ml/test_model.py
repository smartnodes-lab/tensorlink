from src.roles.worker import Worker
from src.roles.validator import Validator
from src.roles.user import User
from src.ml.distributed import DistributedModel
from transformers import Wav2Vec2BertModel, BertModel
import torch.optim as optim
import torch
import json
import time


ip = "127.0.0.1"
port = 5026

mini_batch_size = 1
micro_batch_size = 1


if __name__ == "__main__":
    user = User(
        host=ip,
        port=port,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
        debug=True,
    )
    validator = Validator(
        host=ip,
        port=port + 1,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
        debug=True,
    )
    worker = Worker(
        host=ip,
        port=port + 2,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
        debug=True,
    )

    worker.training = True

    worker.start()
    user.start()
    validator.start()

    worker.connect_dht_node(ip, port + 1)
    user.connect_dht_node(ip, port + 1)

    dummy_input = torch.zeros((1, 2), dtype=torch.long)
    model = BertModel.from_pretrained(
        "bert-base-uncased"
    )  # Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

    d_model = user.request_job(model, 1.4e9)

    # d_model = DistributedModel(model, worker, mini_batch_size, micro_batch_size)
    # d_optimizer = optim.Adam(d_model.parameters(), lr=1e-5)
    # output = d_model(dummy_input)

    # params = d_model.parameters()
    # d_model.train()
    # d_model.eval()

    # losses = [output[o][0].sum() for o in range(mini_batch_size // micro_batch_size)]
    # d_model.backward(losses)
    # d_optimizer.zero_grad()
    # d_optimizer.step()

    worker.stop()
    user.stop()
    validator.stop()
