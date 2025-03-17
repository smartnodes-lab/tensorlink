from tensorlink import DistributedModel, UserNode
from transformers import BertModel
import logging


if __name__ == "__main__":
    model = BertModel.from_pretrained('bert-base-uncased')
    node = UserNode(
        upnp=False, off_chain_test=True, local_test=True, print_level=logging.DEBUG
    )
    distributed_model = DistributedModel(model, node=node)
