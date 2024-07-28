from src.mpc.coordinator import WorkerCoordinator, ValidatorCoordinator, DistributedCoordinator

from transformers import BertModel
import torch
import time


if __name__ == "__main__":

    # user = DistributedCoordinator()
    # time.sleep(0.2)
    worker = WorkerCoordinator()
    time.sleep(0.2)
    validator = ValidatorCoordinator()
    time.sleep(0.2)

    val_key, val_host, val_port = validator.send_request("info", None)

    worker.send_request("connect_node", (val_key, val_host, val_port))

    while True:
        pass

    # user.send_request("connect_node", (val_key, val_host, val_port))
    #
    # model = BertModel.from_pretrained("bert-base-uncased")
    # distributed_model = user.create_distributed_model(model, 1, 1.4e9)
    # din = torch.zeros((1, 1), dtype=torch.long)
    # output = distributed_model(din)
    #
    # print(1)
    #
    # loss = [output[n][0].sum() for n in range(distributed_model.micro_batch_size)]
    # distributed_model.backward(loss)
