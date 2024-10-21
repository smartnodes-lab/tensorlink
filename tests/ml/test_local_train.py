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

    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it",
    #                                           token="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")
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
    #                                              token="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")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    # train(model, tokenizer, device, BATCH_SIZE)
    workers = {
        b'509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f':
            {
                'id': b'509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
                'memory': 2e9,
                'role': b'W',
                'training': True
            },
        b'409d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f':
            {
                'id': b'409d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
                'memory': 2e9,
                'role': b'W',
                'training': True
            },
    }


    def encode_bytes(d):
        """Recursively encode byte keys and byte values to JSON-compatible format."""
        if isinstance(d, dict):
            return {
                key.decode('utf-8') if isinstance(key, bytes) else key: encode_bytes(value)
                for key, value in d.items()
            }
        elif isinstance(d, list):
            return [encode_bytes(item) for item in d]
        elif isinstance(d, bytes):
            return d.decode('utf-8')
        else:
            return d

    def print_indented(model, data):
        print(model.__class__.__name__)

        # Store the previous depth to draw lines correctly
        previous_depth = -1
        parent_names = []  # To track the hierarchy of parent names

        for key, value in data.items():
            # Get the depth of the mod_id
            depth = len(value['mod_id'])

            # Add parent names for current depth
            while previous_depth < depth:
                parent_names.append(value['module'])  # Track the parent module name
                previous_depth += 1

            # Print connecting lines for deeper layers
            if previous_depth > 0:
                print('    ' * (previous_depth - 1) + '|')

            # Print the module name and key on the same line
            indent = '    ' * depth  # Base indentation for the current depth
            print(indent + f"{value['module']}")
            print(indent + f'"{key}": {value["type"]}')

            # Print the parent module names above the current layer
            if previous_depth > 0:
                print(indent + '    ' + ' -> '.join(parent_names[:-1]))

            # Update previous depth
            previous_depth = depth

            # Remove the current module name from parent_names for the next iteration
            if parent_names:
                parent_names.pop()


    # Example usage
    outputs = handle_layers(
        module=model,
        user_memory=int(8e8),
        worker_info=workers,
        handle_layer=False
    )[0]

    print_indented(model, encode_bytes(outputs))
