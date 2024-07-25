from transformers import BertModel
from src.mpc.coordinator import DistributedCoordinator

if __name__ == "__main__":
    user = DistributedCoordinator()

    model = BertModel.from_pretrained("bert-base-uncased")
    user.create_distributed_model(model)
