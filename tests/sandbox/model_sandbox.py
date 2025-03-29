from tensorlink import DistributedModel, UserNode
from transformers import BertModel
import logging


if __name__ == "__main__":
    model = BertModel.from_pretrained('bert-base-uncased')
    distributed_model = DistributedModel(model, node=node)
