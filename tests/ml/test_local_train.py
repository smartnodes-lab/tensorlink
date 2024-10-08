from useful_scripts import *

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('training.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

BATCH_SIZE = 64


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it",
                                              token="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")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
                                                 token="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")

    train(model, tokenizer, device, logger, BATCH_SIZE)
