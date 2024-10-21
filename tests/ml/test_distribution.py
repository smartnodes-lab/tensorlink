import torch
import torch.nn as nn
import multiprocessing
from transformers import AutoTokenizer, AutoModelForCausalLM

from tensorlink.ml.module import DistributedModel


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":

    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it",
    #                                           token="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")
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
    #                                              token="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")
    model = DummyModel()
    dmodel = DistributedModel(q1, q2, lock, model, 1)
    dmodel.distribute_model()
