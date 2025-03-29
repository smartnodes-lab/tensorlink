# Tensorlink
![Static Badge](https://img.shields.io/badge/v0.1.3-Tensorlink-pink?logo=) ![License](https://img.shields.io/badge/License-MIT-blue.svg) ![GitHub Repo stars](https://img.shields.io/github/stars/smartnodes-lab/tensorlink?style=social) 
<!-- ![GitHub all releases](https://img.shields.io/github/downloads/smartnodes-lab/tensorlink/total) -->

**Tensorlink** is a library designed to simplify distributed PyTorch model training and inference, offering tools 
to easily distribute models across a network of peers and share computational resources both locally and globally.

## Peer-to-Peer, Plug-and-Play Distributed Neural Networks for PyTorch

Tensorlink is a versatile framework designed to facilitate neural network offloading and acceleration within PyTorch, a 
leading machine learning framework in Python. Tensorlink offers a straightforward, plug-and-play solution that parses 
and distributes neural networks in PyTorch with ease, including support for third-party models utilized in libraries 
such as Hugging Face. This approach enables the training of large models from consumer hardware, eliminating the need 
for cloud services for certain ML applications. Tensorlink leverages techniques such as automated model parsing and 
parallelism to simplify and enhance the training process, making state-of-the-art models accessible to a wider audience.

For a deeper dive into Tensorlink's features, capabilities, and underlying principles, please refer to the 
[lightpaper](docs/Lightpaper.md) and [documentation](https://smartnodes.ca/docs).

## Table of Contents
1. [Key Features & Limitations](#key-features)
2. [Training & Inference with PyTorch](#training-and-inference-with-pytorch)
3. [Inference APIs](#inference-apis)
4. [Running a Node](#running-a-node)
5. [Contribute](#contributing)

## Key features
By implementing wrappers for PyTorch's `Module` and `Optimizer` objects, Tensorlink integrates with existing codebases, 
preserving model workflows while seamlessly harnessing distributed resources. Tensorlink enables individuals and 
organizations to collaborate, share resources, and scale models dynamically, bringing the power of distributed training 
to a broader community.

- `DistributedModel`: A flexible wrapper for `torch.nn.Module` designed to simplify distributed machine learning workflows.
    - Provides methods for parsing, distributing, and integrating PyTorch models across devices.
    - Supports standard model operations (e.g., `forward`, `backward`, `parameters`).
    - Automatically manages partitioning and synchronization of model components across nodes.
    - Seamlessly supports both data and model parallelism.


- `DistributedOptimizer`: An optimizer wrapper built for `DistributedModel` to ensure synchronized parameter updates across distributed nodes.
   - Compatible with native PyTorch and Hugging Face optimizers.


- **APIs** handling job requests and on-demand inference for open-source huggingface models.

   
- **Public Computational Resources**: By default, Tensorlink nodes are integrated with a smart contract-secured network, enabling:
   - Incentive mechanisms to reward contributors for sharing computational power.
   - Access to both free and paid machine learning resources.
   - Configuration options for private networks, supporting local or closed group machine learning workflows.

### Current Limitations

- Bugs, performance issues, and limited network availability are expected.
- **Model Support**: Tensorlink currently supports open-source Hugging Face models not requiring API-keys. Safe and 
secure custom model distribution methods are currently under development.
- **Model Size Constraints**: Due to limited worker availability in this initial release, public jobs are best suited 
for models under ~10 billion parameters.
- **Worker Allocation**: Public jobs are currently limited to one worker. Data parallel acceleration is temporarily 
disabled for public tasks but can be enabled for local jobs or private clusters.
- Internet latency and connection speeds can significantly impact the performance of public jobs, which may become 
problematic for certain training and inference scenarios.


## Training and Inference with PyTorch

### Installation

Before installing Tensorlink, ensure you meet the following requirements:

- UNIX/MacOS (Windows support coming soon...)
- Python 3.11.9+
- PyTorch 2.3+ (ensure model compatibility with torch version)

While we aim to reduce version requirements, Python 3.11.9+ and a UNIX-based OS are required for stable utilization. 
You can install Tensorlink using pip.

```bash
pip install tensorlink
```

This command will download and install Tensorlink along with its dependencies. If you're using a virtual environment 
(recommended), ensure it's activated before running the installation command.

*Tensorlink aims to be compatible with all models and optimizers built with of PyTorch, however some compatibility 
issues can be expected with the pre-alpha release.* 

To get started you must request a job. Requesting a job will 
provide you with a distributed model and optimizer objects. The optimizer must be instantiated with kwargs after the 
request of a job, leaving out model parameters. When requesting a job, ensure that the request follows the 
instantiation of your model and precedes the training segment of your code:

```python
from tensorlink import UserNode
from transformers import AutoModelForCausalLM
from torch.optim import AdamW

# Initialize tokenizer, model, optimizer, and loss function
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")

# Create a Tensorlink user node instance, and request a job with your model
user = UserNode()
distributed_model, distributed_optimizer = user.create_distributed_model(
      model=model,
      training=True,
      optimizer_type=AdamW,
)
distributed_optimizer(lr=5e-5) # Instantiate optimizer without specifying parameters
```

Once the job request is created, you'll be successfully connected to Tensorlink. You can now proceed with training. 
Here’s an example of a training loop that uses the distributed model:

```python
from torch.utils.data import DataLoader

# Training loop
epochs = 10
for epoch in range(epochs):
    # Iterating over tokenized dataset. See tests/ml/useful_scripts.py
    for batch in DataLoader(tokenized_dataset["train"], batch_size=8):
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['label'].to(device)

        distributed_optimizer.zero_grad()
        outputs = distributed_model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        distributed_optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} completed")
```

Training progress and network information will be trackable through the Tensorlink/Smartnodes dashboard. 
This feature is a work in progress and is currently not available.


## Inference APIs


## Running a Node

Tensorlink is a versatile system designed for use on local, private, and public networks. However, its true power shines on the **public network**, where individuals worldwide contribute computational resources to advance innovation. By running a Worker node, you not only support cutting-edge projects but also earn rewards for your contributions.

### Why Run a Tensorlink Node?
- **Support Innovation**: Contribute to global machine learning and computational projects.
- **Earn Rewards**: Get compensated for providing your idle GPU power.
- **Join the Community**: Be part of a decentralized network pushing the boundaries of technology.

### Getting Started

1. **Download the Node Binary**  
   - Visit the [**Releases**](https://github.com/smartnodes-lab/tensorlink) section on GitHub to download the latest `tensorlink-miner` binary for your platform.
      - Multi-GPU utilization and Windows support are not yet supported.
   - Ensure you have Python 3 and a **CUDA-enabled GPU** installed on your system.

2. **Set Up the Configuration**  
   - Open the `config.json` file.
   - Add:
     - Your **base wallet address** (used for receiving rewards).
     - The **path to a GPU mining script or other process** you wish to run while the worker is idle (optional).
         - You must also set "mining" to "true".

3. **Run the Worker**  
   - Execute the *run-worker.sh* script to start your node. (e.g. `./run-worker.sh`)


## Contributing

We’re excited to welcome contributions from the community to help build and enhance Tensorlink! Here’s how you can get involved:

- **Report Issues:** Encounter a bug or have a feature request? Create an issue on our GitHub repository.
- **Submit Pull Requests:** Fork the repository, make improvements or fixes, and send us a pull request.
- **Documentation Contributions:** Help improve the Tensorlink Docs.
- **Join the Discussion:** Connect with us and other contributors on our Discord server.

We need more people to help us refine Tensorlink and make it the best possible tool for decentralized neural network training. Your contributions and insights can make a significant impact!

### Donate

If you would like to support our work, consider buying us a coffee! Your contributions help us continue developing and improving Tensorlink.

<a href="https://www.buymeacoffee.com/smartnodes" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="width: 150px; height: auto;">
</a>
