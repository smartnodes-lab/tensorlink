import torch.nn as nn
import subprocess


def get_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode == 0:
            # Extract the GPU memory total from the output
            gpu_memory_total = int(result.stdout.strip())
            return gpu_memory_total

        else:
            print(f"Error running nvidia-smi: {result.stderr}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def get_devices():
    # Query network for
    pass


def parse_model(model):
    pass


if __name__ == '__main__':
    gpu_memory_total = get_gpu_memory()

    if gpu_memory_total is not None:
        print(f"Total GPU Memory: {gpu_memory_total} MB")
