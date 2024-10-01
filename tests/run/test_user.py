from transformers import BertModel
from src.mpc.coordinator import DistributedCoordinator

if __name__ == "__main__":
    user = DistributedCoordinator()
    # user.send_request("connect_node", (b"", "jumbomeats.ddns.net", 5026))
    #
    # model = BertModel.from_pretrained("bert-base-uncased")
    # user.create_distributed_model(model)
