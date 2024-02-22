from src.ml.distributed import DistributedModel, print_distribute_model
from src.ml.model_analyzer import handle_output
from src.ml.worker import Worker

from transformers import BertModel, Wav2Vec2BertModel
import torch.nn as nn
import torch
import time
import os


if __name__ == "__main__":

    dummy_input = torch.zeros((1, 1), dtype=torch.long)
    model = BertModel.from_pretrained("bert-base-uncased")
    # model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

    a = next(model.children())(dummy_input)

    print_distribute_model(model)
