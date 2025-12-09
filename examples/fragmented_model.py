"""Test the inference of a tiny model across two local worker nodes"""

import json

from tensorlink.ml.graphing import ModelParser

parser = ModelParser(verbose=True)
config = parser.create_distributed_config(
    "Qwen/Qwen3-8B",
    {
        '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
            'id': '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
            'gpu_memory': 16e9,
            'total_gpu_memory': 16e9,
            'role': 'W',
            'training': False,
        },
        '209d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
            'id': '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
            'gpu_memory': 16e9,
            'total_gpu_memory': 16e9,
            'role': 'W',
            'training': False,
        },
    },
    False,
    False,
    False,
    False,
    max_seq_len=4096,
    batch_size=4,
)


config
