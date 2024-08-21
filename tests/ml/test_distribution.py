import torch
import torch.nn as nn
import multiprocessing
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.ml.distributed import DistributedModel

q1 = multiprocessing.Queue()
q2 = multiprocessing.Queue()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it",
                                          token="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")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
                                             token="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")

dmodel = DistributedModel(q1, q2, model, 1)
dmodel.worker_info = {
    b'509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
        'id': b'509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f', 'mpc': 8e9, 'role': b'W',
        'training': True},
    b'409d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
        'id': b'409d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f', 'mpc': 24e9, 'role': b'W',
        'training': True}
}
dmodel.parse_model(model)
