<p align="center">
  <img src="https://raw.githubusercontent.com/smartnodes-lab/tensorlink/main/docs/images/logo.png" alt="Logo" width="400" style="max-width:100%; border-radius:12px;">
</p>

<h3 align="center">Distributed AI Inference & Training for Everyone</h3>

<p align="center">
 <img src="https://img.shields.io/github/v/release/smartnodes-lab/tensorlink?label=Latest%20Release&color=ff69b4" alt="Latest Release Version" />
  <img src="https://img.shields.io/github/downloads/smartnodes-lab/tensorlink/total?label=Node%20Downloads&color=e5e52e" alt="Node Downloads"/>
  <img src="https://img.shields.io/github/stars/smartnodes-lab/tensorlink?style=social" alt="GitHub Repo stars"/>
  <a href="https://discord.gg/aCW2kTNzJ2">
    <img src="https://img.shields.io/badge/Join%20Discord-5865F2?logo=discord&logoColor=white" alt="Join us on Discord"/>
  </a>
  <a href="https://smartnodes.ca/tensorlink/docs" target="_blank">
    <img src="https://img.shields.io/badge/Documentation-1d72b8?logo=readthedocs&logoColor=white" alt="Documentation"/>
  </a>
</p>

## What is Tensorlink?

Tensorlink is a Python library and decentralized platform that makes distributed AI accessible to everyone. 
Access Hugging Face models through simple APIs, run PyTorch models across a network of peers, or contribute compute 
resources to earn rewards, all without the need of centralized infrastructure.

> **Early Access:** We're in active development! Some features are still stabilizing. [Join our Discord](https://discord.gg/aCW2kTNzJ2) for updates and support.

### Key Features
- **Drop-in PyTorch replacement** - Run models in your workflows without VRAM
- **Simple REST APIs** - Access Hugging Face models with familiar HTTP requests  
- **Privacy-first architecture** - Your data stays local, never stored on external servers
- **Earn while you contribute** - Get rewarded for sharing idle compute resources

## Quick Start

Tensorlink can be accessed via API or directly within Python. 

### Use the Inference API

```python
import requests

response = requests.post(
    "http://smartnodes-lab.ddns.net/tensorlink-api/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "Explain quantum computing in simple terms",
        "max_new_tokens": 256,
        "temperature": 0.7
    }
)

print(response.json())
```

### Installation

```bash
pip install tensorlink
```

**Requirements:** Python 3.10+, PyTorch 2.3+, UNIX/MacOS (Windows support coming soon)

### Run Your First Distributed Model

```python
from tensorlink import DistributedModel
import torch

# Connect to a pre-trained model on the network
model = DistributedModel(
    model="Qwen/Qwen2.5-7B-Instruct",
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

- 🐛 **Report bugs** via [GitHub Issues](https://github.com/smartnodes-lab/tensorlink/issues)
- 💡 **Suggest features** on our [Discord](https://discord.gg/aCW2kTNzJ2)
- 🔧 **Submit PRs** to improve code or documentation
- ☕ **Support the project** via [Buy Me a Coffee](https://www.buymeacoffee.com/smartnodes)

## License

Tensorlink is released under the [MIT License](LICENSE).
