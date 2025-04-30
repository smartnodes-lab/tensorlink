<div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; max-width: 400px; width: 100%; margin-left: auto; margin-right: auto;">
  <img src="docs/images/logo_dark.png" alt="Logo" style="max-width: 100%; margin-bottom: 16px; border-radius: 12px;">
</div>

<h3 align="center" style="margin-top: -32px">Distributed AI Inference & Training for Everyone</h3>
<p align="center">
  <i>Plug-and-play models and APIs for distributed neural network inference and training with PyTorch and Hugging Face.</i>
</p>
<p align="center">
  <img src="https://img.shields.io/github/v/release/smartnodes-lab/tensorlink?label=Latest%20Release&color=ff69b4" alt="Latest Release Version" />
  <img src="https://img.shields.io/github/release-date/smartnodes-lab/tensorlink?color=lightgrey&label=Release%20Date" alt="Release Date" />
  <img src="https://img.shields.io/github/downloads/smartnodes-lab/tensorlink/total?label=Node%20Downloads&color=e5e52e" alt="Node Downloads"/>
  <img src="https://img.shields.io/github/stars/smartnodes-lab/tensorlink?style=social" alt="GitHub Repo stars"/>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"/>
  <a href="https://discord.gg/aCW2kTNzJ2">
    <img src="https://img.shields.io/badge/Join%20Discord-5865F2?logo=discord&logoColor=white" alt="Join us on Discord"/>
  </a>
</p>


> ⚠️ **Pre-release Notice:** This is an early version of the project. Some features may be incomplete or unstable. **Not 
> recommended for production use at this time.**

**Tensorlink** is a Python library and computational platform that provides powerful tools and APIs for large-scale 
neural network training and inference in PyTorch. It enables users to work with complex models that exceed the memory 
limits of a single device, expanding access to cutting-edge deep learning. Tensorlink streamlines the parsing and 
distribution of models, and provides a framework for accessing and sharing computation directly peer-to-peer, making 
powerful models available on demand.

## Table of Contents
1. [Introduction & Key Features](#introduction)
2. [Training & Inference with PyTorch](#training-and-inference-with-pytorch)
3. [Inference APIs](#inference-apis)
4. [Running a Node](#running-a-node)
5. [Utilizing Local & Private Devices]()
6. [Contribute](#contributing)

> 💡 **Looking to get started?** Jump to [Training & Inference with PyTorch](#training-and-inference-with-pytorch) for a hands-on guide to running your first distributed model with Tensorlink.

> 🖥️ **Interested in Powering the Network?** Learn how in the [Running a Node](#running-a-node) section to set up your own node and join the network.

---

## Introduction

Tensorlink is a flexible and powerful framework designed to facilitate neural network offloading and acceleration within 
PyTorch, a leading machine learning framework in Python. It simplifies the parsing and distribution of models, supporting
pre-trained architectures from libraries like Hugging Face, enabling seamless execution across distributed consumer 
hardware. By leveraging techniques such as model sharding, parallel workflow execution, automated peer discovery, and a 
built-in incentive system, Tensorlink provides an efficient, decentralized alternative to traditional cloud-based ML 
services. This significantly lowers the barrier to entry for both training and inference, empowering individuals and 
organizations to deploy state-of-the-art AI models without the need for costly, centralized infrastructure.

### Key Features

Tensorlink integrates directly into PyTorch codebases through lightweight wrappers around core PyTorch objects such as 
`Module` and `Optimizer`. This allows developers to maintain familiar workflows while scaling models dynamically across 
a distributed compute network. By enabling collaboration and resource-sharing between users, Tensorlink brings the power
of distributed training and inference to a broader community.

### `DistributedModel`
A wrapper around `torch.nn.Module` objects designed to simplify the process of running models across multiple devices
or nodes. It automatically parses and distributes model submodules across worker nodes, making efficient use of 
available compute. Crucially, it preserves the standard PyTorch interface, including `forward`, 
`backward`, and `parameters` — allowing developers to integrate it into existing codebases with minimal friction. 
Tensorlink supports both model parallelism and data parallelism, and handles synchronization and communication between
distributed components behind the scenes, streamlining complex workflows.

### `DistributedOptimizer`
The `DistributedOptimizer` is built to complement `DistributedModel`, providing synchronized parameter updates across 
distributed training nodes. It is fully compatible with PyTorch’s built-in optimizers as well as third-party optimizers 
used in Hugging Face transformers. This ensures seamless integration into diverse training pipelines and guarantees 
consistent updates in sharded or parallelized model training environments, improving training stability and 
reproducibility in distributed contexts.

### On-Demand Inference APIs
Tensorlink includes an API for on-demand inference using open-source Hugging Face pre-trained models. These APIs 
allow users to instantly access popular models in their applications.

### Public & Private Compute Networks
By default, all Tensorlink nodes are connected through a smart contract-secured peer-to-peer mesh. This decentralized 
architecture enables users to share their idle computational resources and earn token-based rewards in return. The 
network supports both free and paid usage of resources, giving users flexible options depending on their compute needs and 
budget.


### ⚠️ Current Limitations
As Tensorlink is still in its early release phase, users may encounter bugs, performance inconsistencies, and limited
network availability. Currently, model support is focused on open-source Hugging Face models that do not require API 
keys. Safe and secure methods for custom model distribution are under active development and will be available in future 
updates.

In this early stage, there are also some practical constraints related to model size and resource allocation. Due to 
limited availability of public workers, tasks involving models larger than approximately 10 billion parameters may not 
perform optimally. Additionally, public inference and training jobs are currently restricted to a single worker, with 
data parallelism temporarily disabled for these tasks. However, data parallel acceleration remains available for local 
jobs and within private clusters.

Finally, internet latency and connection quality can significantly affect performance for public tasks. This may pose 
challenges for latency-sensitive or high-throughput training and inference scenarios. As the network matures, these 
limitations are expected to be progressively addressed.

---

## Training and Inference with PyTorch

### Installation

Before installing Tensorlink, ensure you meet the following requirements:

- UNIX/MacOS (Windows support coming soon...)
- Python 3.10+
- PyTorch 2.3+ (ensure model compatibility with torch version)

While version constraints will be relaxed in future releases, Python 3.10+ and a UNIX-based OS are currently required for stable usage.

To install Tensorlink, simply use pip:

```bash
pip install tensorlink
```

This command will install Tensorlink and all its dependencies. If you're working in a virtual environment (recommended), make sure it's activated before installing.

*⚠️ Tensorlink is designed to be compatible with all PyTorch-based models and optimizers. However, some issues can be
expected to occur during the pre-alpha phase.*

### Creating a Distributed Model

A `DistributedModel` is a wrapper that automatically connects your machine to the Tensorlink network and offloads your
model to available Workers. It behaves like a standard PyTorch model and supports three ways to define the model:

- A Hugging Face model name (e.g. `"microsoft/microsoft-Phi-4B-Instruct"`)
- A custom `torch.nn.Module` object
- A local file path to saved model parameters (`.pt` or `.bin`)

You can also use the distributed model to spawn an optimizer using `DistributedModel.create_optimizer`, which handles remote synchronization automatically.

```python
from tensorlink import DistributedModel
from torch.optim import AdamW
from my_custom_model import CustomModel  # Optional: Your custom model

# Option 1: Hugging Face model (Stable)
distributed_model = DistributedModel(
    model="Qwen/Qwen2.5-7B-Instruct",
    training=False
)

# Option 2: Custom PyTorch model (⚠️ Experimental — Under development, may not work as expected)
distributed_model = DistributedModel(
    model=CustomModel(),
    training=True,
    optimizer_type=AdamW
)

# Option 3: Load from local parameters file (⚠️ Experimental — Under development, support is incomplete)
# distributed_model = DistributedModel(
#     model="path/to/model_weights.pt",  # or .bin
#     training=False,
#     optimizer_type=AdamW
# )

# Create optimizer (only needed for training)
distributed_model.create_optimizer(lr=5e-5)

```

Training progress and network activity will soon be viewable through the [Smartnodes](https://smartnodes.ca/app) dashboard (currently under development).

---

## Inference APIs

Tensorlink offers a lightweight API for performing distributed inference, allowing access to popular 
Hugging Face pre-trained models on-demand. Furthermore, you may offload your model using the `DistributedModel` and call
it just like a regular PyTorch model—whether from a local script or remotely. 

### Exmples

#### Python (with `requests`)

```python
import requests

https_serv = "https://smartnodes-lab.ddns.net/tensorlink-api"  # May not work with all clients 
http_serv = "http://smartnodes-lab.ddns.net:443/tensorlink-api"  # Use this if HTTPS fails

payload = {
    "hf_name": "Qwen/Qwen2.5-7B-Instruct",
    "message": "Describe the role of AI in medicine.",
    "max_length": 1024,
    "max_new_tokens": 256,
    "temperature": 0.7,
    "do_sample": True,
    "num_beams": 4,
    "history": [
        {"role": "user", "content": "What is artificial intelligence?"},
        {"role": "assistant", "content": "Artificial intelligence refers to..."}
    ]
}

response = requests.post(f"{http_serv}/generate", json=payload)
print(response.json())
```


#### JavaScript / TypeScript (Fetch API)

```js
// Available endpoints (status may vary):
const https_serv = "https://smartnodes-lab.ddns.net/tensorlink-api";  // May not work with all clients
const http_serv = "http://smartnodes-lab.ddns.net:443/tensorlink-api"; // Use this if HTTPS fails

const response = await fetch(http_serv + '/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    hf_name: modelParams.model,
    message: userMessage.content,
    max_length: modelParams.maxLength,
    max_new_tokens: modelParams.maxNewTokens,
    temperature: modelParams.temperature,
    do_sample: modelParams.doSample,
    num_beams: modelParams.numBeams,
    history: messages.map(msg => ({ 
      role: msg.role, 
      content: msg.content 
    })),
  }),
});

const result = await response.json();
console.log(result);

```

### 📥 API Parameters

| Field            | Type    | Required | Description                                            |
|------------------|---------|----------|--------------------------------------------------------|
| `hf_name`        | string  | ✓        | Name of the Hugging Face model                         |
| `message`        | string  | ✓        | The user's input prompt or question                    |
| `max_length`     | int     | ✕        | Total token limit (input + output)                     |
| `max_new_tokens` | int     | ✕        | Maximum number of tokens to generate                   |
| `temperature`    | float   | ✕        | Sampling temperature (e.g., `0.7` = more creative)     |
| `do_sample`      | boolean | ✕        | Whether to sample (`True`) or use greedy decoding      |
| `num_beams`      | int     | ✕        | Beam search width (`1` for greedy, `>1` for diversity) |
| `history`        | array   | ✕        | Conversation history (`[{ role, content }]`)           |


### ⚠️ Note
- Currently limited to select HF models (listed in `tensorlink/ml/models.json`)
  - Custom models and more diverse selection coming soon...
- Keep histories concise for faster response time.
- Model loading and generation performance depends on network conditions and node availability.

---

### Utilizing Local & Private Devices

While the public Smartnodes network is designed for distributed AI workloads, certain use cases require higher levels of privacy, data control, or hardware isolation. **Smartnodes also supports fully private or LAN-based deployments** on your own hardware, ideal for running sensitive training or inference jobs.

#### Setup Instructions
1. **Disable P2P Discovery**: Set `peer_discovery = False` in your node configuration to prevent broadcasting your presence.
2. **Custom Peering**: Manually define trusted local peers via IP and port, creating a closed loop of devices under your control.
3. **Storage & Models**: Store models and shared memory on a centralized NAS or local SSD to reduce latency.
4. **Security**: Use firewall rules and VLAN segmentation for network isolation. Load private keys from hardware wallets or encrypted vaults.

### Creating a Private AI Cluster

For users looking to build a **mini AI data center** or test Tensorlink functionality in an isolated environment, the following example demonstrates how to set up a fully private network using local-only devices:

```python
from tensorlink import UserNode, ValidatorNode, WorkerNode, DistributedModel
import torch, logging, time
from transformers import AutoTokenizer

# Local setup parameters
LOCAL = True          # Force localhost-only connections (127.0.0.1)
UPNP = not LOCAL      # Disable UPnP to prevent external exposure
OFFCHAIN = LOCAL      # Use off-chain job coordination (fully private)

model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

# On Device 1
validator = ValidatorNode(upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG)
# On Device 2
user = UserNode(upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG)
# On Device 3
worker = WorkerNode(upnp=UPNP, off_chain_test=OFFCHAIN, local_test=LOCAL, print_level=logging.DEBUG)

# Connect worker and user to validator manually
val_key, val_host, val_port = validator.send_request("info", None)
time.sleep(1)
worker.connect_node(val_host, val_port, node_id=val_key)
time.sleep(1)
user.connect_node(val_host, val_port, node_id=val_key)
time.sleep(1)

# Request a distributed inference model
distributed_model = DistributedModel(model_name, training=False, node=user)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Perform local inference loop
for _ in range(5):
    input_text = "You: Hello Bot."
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = distributed_model.generate(
            inputs,
            max_new_tokens=256,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    print("Bot:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# Shutdown
user.cleanup()
worker.cleanup()
validator.cleanup()
```

#### Notes:
- All nodes are simulated on the same machine or LAN group.
- Customize `connect_node()` with local IPs to run across multiple physical devices on a WAN/LAN.

---

## Running a Node

Tensorlink is designed to work across **local**, **private**, and **public** networks—but the public network is where it truly comes alive. By joining the decentralized ecosystem, your machine becomes part of a global infrastructure powering machine learning applications. Whether you're a hobbyist or a data center operator, running a Tensorlink node earns you rewards and directly contributes to the future of AI.

### Why Run a Tensorlink Node?
- 🚀 **Support Innovation**: Lend your GPU power to cutting-edge research and open-source projects.
- 💸 **Earn Rewards**: Get compensated for your compute time—idle GPUs become productive assets.
- 🌐 **Join the Movement**: Help build a censorship-resistant, decentralized compute backbone.

### Getting Started

1. **Download the Node Binary**  
   - Grab the latest `tensorlink-miner` from the [**Releases**](https://github.com/smartnodes-lab/tensorlink) page.
   - Make sure your system has:
     - Python 3
     - A **CUDA-enabled GPU**

2. **Configure Your Node**  
   - Open the `config.json` file and set:
     - `"wallet"`: Your Ethereum-compatible wallet address (for receiving rewards).
     - `"mining"`: Set to `true` if you want to run a local script while idle.
     - `"mining_script"`: (Optional, BROKEN) Path to the script you want to run when not handling jobs.

3. **Run the Worker**  
   - Launch your node using the provided script:
     ```bash
     ./run-worker.sh
     ```

   - You should start seeing logs that indicate connection to the network and readiness to receive jobs.

---


## Contributing

Contributions to help build and improve Tensorlink are always welcome! Here's how you can get involved:

- **Report Issues:** If you encounter a bug or have a feature suggestion, please create an issue on our [GitHub repository](#).
- **Submit Pull Requests:** Fork the repository, implement improvements or fixes, and submit a pull request.
- **Contribute to Documentation:** Help enhance the [Tensorlink Docs](#) to make it more user-friendly and comprehensive.
- **Join the Community:** Connect with us and other contributors on our [Discord server](#) to share ideas, ask questions, or collaborate.

Your contributions, whether through code, feedback, or documentation, are essential in making Tensorlink the best tool 
for decentralized neural network training. We appreciate your help!

### Donate

If you would like to support our work, consider buying us a coffee! Your contributions help us continue developing and improving Tensorlink.

<a href="https://www.buymeacoffee.com/smartnodes" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="width: 150px; height: auto;">
</a>
