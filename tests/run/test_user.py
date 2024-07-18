from transformers import BertModel
from src.coordinator import DistributedCoordinator

import time


if __name__ == "__main__":
    user = DistributedCoordinator()

    model = BertModel.from_pretrained("bert-base-uncased")
    user.create_distributed_model(model)
