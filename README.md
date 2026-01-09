<p align="center">
  <img src="https://raw.githubusercontent.com/mattjhawken/tensorlink/main/docs/images/logo.png" alt="Logo" width="400" style="max-width:100%; border-radius:12px;">
</p>

<h3 align="center">Peer-to-peer AI Inference & Distributed Execution with PyTorch</h3>

<p align="center">
 <img src="https://img.shields.io/github/v/release/mattjhawken/tensorlink?label=Latest%20Release&color=ff69b4" alt="Latest Release Version" />
  <img src="https://img.shields.io/github/downloads/mattjhawken/tensorlink/total?label=Node%20Downloads&color=e5e52e" alt="Node Downloads"/>
  <img src="https://img.shields.io/github/stars/mattjhawken/tensorlink?style=social" alt="GitHub Repo stars"/>
  <a href="https://discord.gg/aCW2kTNzJ2">
    <img src="https://img.shields.io/badge/Join%20Discord-5865F2?logo=discord&logoColor=white" alt="Join us on Discord"/>
  </a>
  <a href="https://smartnodes.ca/tensorlink/docs" target="_blank">
    <img src="https://img.shields.io/badge/Documentation-1d72b8?logo=readthedocs&logoColor=white" alt="Documentation"/>
  </a>
</p>

## What is Tensorlink?

Tensorlink is a Python library and decentralized compute platform for running PyTorch and Hugging Face models across a 
peer-to-peer network of GPUs. It enables:
- Running large models without local VRAM 
- Hosting models on your own hardware and accessing them remotely via API 
- Distributing model execution across multiple consumer GPUs 
- Contributing idle compute to earn network rewards

All without relying on centralized cloud inference providers.

> **Early Access:** Tensorlink is under active development. APIs and internals may evolve. [Join our Discord](https://discord.gg/aCW2kTNzJ2) for updates, support, and roadmap discussions.

### Key Features
- **Native PyTorch Integration** - Wrap Hugging Face or custom PyTorch models and execute them across the network.
- **REST API for Inference** - Access hosted models via HTTP without PyTorch dependencies.  
- **Distributed Model Execution** - Run models larger than a single GPU by partitioning execution across peers.
- **Privacy Options**: Route queries exclusively to your own hardware for private usage.
- **Incentivized Compute Sharing** - Earn rewards by contributing idle GPUs to the network.

## Quick Start

Tensorlink can be accessed via API or directly within Python. 

### Use the Inference API

```python
import requests

inference_query = requests.post(
    "http://smartnodes.ddns.net/tensorlink-api/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "Does this user query require an internet search? Response with either only yes, or no.",
        "max_new_tokens": 32,
        "stream": False,
    }
)


request_model = requests.post(
    "http://smartnodes.ddns.net/tensorlink-api/request-model",
    json={}
)

```

### Installation

```bash
pip install tensorlink
```

**Requirements:** Python 3.10+, PyTorch 2.3+, UNIX/MacOS (Windows support coming soon)

### Run Your First Distributed Model

This example illustrates how to spawn a HuggingFace Pre-trained model on the tensorlink public network. If you  wish to 
leverage your own hardware, or for a more complex breakdown, proceed to the Examples section.

```python
from tensorlink import DistributedModel
import torch

# Connect to a pre-trained model on the network
model = DistributedModel(
    model="Qwen/Qwen3-8B-Instruct",
    training=False,
    device="cuda",
    dtype=torch.float16
)

# Use it like any PyTorch model
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100)
```

### Contribute Compute (Mining)

1. Download the latest `tensorlink-miner` from [Releases](...)
2. Configure your wallet address in `config.json`
3. Run: `./run-worker.sh`

That's it! Your GPU will earn rewards by processing AI workloads from the network.

## Learn More

- 📚 **[Documentation](https://smartnodes.ca/tensorlink/docs)** - Full API reference and guides
- 💬 **[Discord Community](https://discord.gg/aCW2kTNzJ2)** - Get help and connect with developers
- 🎮 **[Live Demo](https://smartnodes.ca/localhostGPT)** - Try localhostGPT powered by Tensorlink
- 📘 **[Litepaper](docs/LITEPAPER.md)** - Technical overview and architecture

## Contributing

We welcome contributions! Here's how to get involved:

- 🐛 **Report bugs** via [GitHub Issues](https://github.com/mattjhawken/tensorlink/issues)
- 💡 **Suggest features** on our [Discord](https://discord.gg/aCW2kTNzJ2)
- 🔧 **Submit PRs** to improve code or documentation
- ☕ **Support the project** via [Buy Me a Coffee](https://www.buymeacoffee.com/smartnodes)

## License

Tensorlink is released under the [MIT License](LICENSE).
