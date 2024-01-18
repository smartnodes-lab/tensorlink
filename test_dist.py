import torch.nn as nn
import torch
from transformers import BertModel


model = BertModel.from_pretrained("bert-base-uncased")
dummy_input = torch.zeros((1, 1), dtype=torch.long)
out = model(dummy_input)


def print_submodules(module: nn.Module):
    for submodule in module.children():

