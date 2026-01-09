"""Test the inference of a tiny model across two local worker nodes"""

from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.utils import estimate_memory, format_memory_size
import pandas as pd

models = [
    # "HuggingFaceTB/SmolLM-135M",
    # "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]

batch_sizes = [1]
seq_lengths = [1024, 4096, 8196]

workers = {
    '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
        'id': '509d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
        'gpu_memory': 24e9,
        'total_gpu_memory': 24e9,
        'role': 'W',
        'training': False,
    },
    '209d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f': {
        'id': '209d89bf56704c67873c328e4f706a705b2fdc1671ebacab1083c9c6d2df650f',
        'gpu_memory': 16e9,
        'total_gpu_memory': 16e9,
        'role': 'W',
        'training': False,
    },
}

# Collect rows for the final DataFrame
rows = []

for model_name in models:
    for bs in batch_sizes:
        for seqlen in seq_lengths:
            parser = ModelParser(verbose=False)

            try:
                config = parser.create_distributed_config(
                    model_name,
                    workers,
                    training=False,
                    trusted=False,
                    handle_layers=False,
                    input_obfuscation=False,
                    host_load_small=True,
                    host_threshold_mb=50,
                    max_seq_len=seqlen,
                    batch_size=bs,
                    optimizer_type="adam",
                )

                success = config.get("success", False)
                model_memory = (
                    format_memory_size(config["model_memory"]) if success else 0
                )
                components_memory = {
                    k: n['memory'] for k, n in config["config"].items()
                }

                # Append to rows
                rows.append(
                    {
                        "model": model_name,
                        "batch_size": bs,
                        "seq_length": seqlen,
                        "model_memory": model_memory,
                        "components_sum": format_memory_size(
                            sum(list(components_memory.values()))
                        ),
                        "components_memory": {
                            k: format_memory_size(v)
                            for k, v in components_memory.items()
                        },
                        "success": success,
                        "error": None if success else config.get("error", None),
                    }
                )

            except Exception as e:
                rows.append(
                    {
                        "model": model_name,
                        "batch_size": bs,
                        "seq_length": seqlen,
                        "success": False,
                        "error": str(e),
                    }
                )

# Create a pandas DataFrame
df = pd.DataFrame(rows)

print("\n\n========== FINAL RESULTS TABLE ==========\n")
print(df.to_string(index=False))
print("\nTable stored in variable: df")
