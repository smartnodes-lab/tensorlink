from tensorlink.ml.graphing import *
from tensorlink.ml.module import DistributedModel
from useful_scripts import *
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging

BATCH_SIZE = 64

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('training.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    train(model, tokenizer, device, BATCH_SIZE)
