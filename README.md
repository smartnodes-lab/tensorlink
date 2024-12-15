# Tensorlink
![Static Badge](https://img.shields.io/badge/v0.1.0-Tensorlink-pink?logo=) ![License](https://img.shields.io/badge/License-MIT-blue.svg) ![GitHub Repo stars](https://img.shields.io/github/stars/smartnodes-lab/tensorlink?style=social) 
<!-- ![GitHub all releases](https://img.shields.io/github/downloads/smartnodes-lab/tensorlink/total) -->

**Tensorlink** is a library designed to simplify the scaling of PyTorch model training and inference, offering tools 
to easily distribute models across a network of peers and share computational resources both locally and globally.

## Plug-and-Play, Peer-to-Peer Neural Network Scaling for PyTorch

Tensorlink is a versatile framework designed to facilitate neural network offloading and acceleration within PyTorch, a 
leading machine learning framework in Python. Tensorlink offers a straightforward, plug-and-play solution that parses 
and distributes neural networks in PyTorch with ease, including support for third-party models utilized in libraries 
such as Hugging Face. This approach enables the training of large models from consumer hardware, eliminating the need 
for cloud services for certain ML applications. Tensorlink leverages techniques such as automated model parsing and 
parallelism to simplify and enhance the training process, making state-of-the-art models accessible to a wider audience.

For a deeper dive into Tensorlink's features, capabilities, and underlying principles, please refer to the 
[lightpaper](docs/Lightpaper.md) and [documentation](https://smartnodes.ca/docs).

### Key features
By implementing wrappers for PyTorch's `Module` and `Optimizer` objects, Tensorlink integrates with existing codebases, 
preserving model workflows while seamlessly harnessing distributed resources. Tensorlink enables individuals and 
organizations to collaborate, share resources, and scale models dynamically—bringing the power of distributed training 
to a broader community.

- `DistributedModel`: A flexible wrapper for `torch.nn.Module` designed to simplify distributed machine learning workflows.
    - Provides methods for parsing, distributing, and integrating PyTorch models across devices.
    - Supports standard model operations (e.g., `forward`, `backward`, `parameters`).
    - Automatically manages partitioning and synchronization of model components across nodes.
    - Seamlessly supports both data and model parallelism.

- `DistributedOptimizer`: An optimizer wrapper built for `DistributedModel` to ensure synchronized parameter updates across distributed nodes.
   - Compatible with native PyTorch and Hugging Face optimizers.

- Nodes Types (`tensorlink.nodes`): Tensorlink provides three key node types to enable robust distributed machine learning workflows:
   - `UserNode`: Handles job submissions and result retrieval, facilitating interaction with `DistributedModel` for training and inference. Required for public network participation.
   - `WorkerNode`: Manages active jobs, connections to users, and processes data for model execution.
   - `ValidatorNode`: Secures and coordinates training tasks and node interactions, ensuring job integrity on the public network.
   
- **Public Computational Resources**: By default, Tensorlink nodes are integrated with a smart contract-secured network, enabling:
   - Incentive mechanisms to reward contributors for sharing computational power.
   - Access to both free and paid machine learning resources.
   - Configuration options for private networks, supporting local or closed group machine learning workflows.

### Limitations in this Release

- Bugs, performance issues, and limited network availability are expected.
- **Model Support**: Tensorlink currently supports scriptable PyTorch models (`torch.jit.script`) and select open-source 
Hugging Face models not requiring API-keys.
   - **Why?** Security and serialization constraints for un-trusted P2P interactions. We're actively working on custom serialization methods to support all PyTorch model types. Feedback and contributions to accelerate this effort are welcome!
- **Job Constraints**: 
    - **Model Size**: Due to limited worker availability in this initial release, public jobs are best suited for models under ~1 billion parameters.
        - **Future Plans**: We are actively expanding network capacity, and the next update (expected soon) will increase this limit, enabling support for larger models and more complex workflows.
    - **Worker Allocation**: Public jobs are currently limited to one worker. Data parallel acceleration is temporarily disabled for public tasks but can be enabled for local jobs or private clusters.
- Internet latency and connection speeds can significantly impact the performance of public jobs, which may become problematic for certain training and inference scenarios.


## Training and Inference with Tensorlink

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

## Running a Node

Tensorlink can be configured for use on local or private networks, but its full potential lies in the public network, 
where individuals from around the world contribute computational resources. By running a Worker node, you can:

- **Support Innovation:** Contribute to building a global decentralized compute network.
- **Earn Rewards:** Provide resources and receive Smartnodes tokens (SNO) for your contributions.
- **Join the Community:** Be part of an open-source project aiming to redefine distributed computing.


### How to Get Started
- Check the **Releases** section on GitHub for binaries or scripts to set up a node quickly and easily.
- Follow the included documentation to configure your node and start contributing to the Tensorlink network.

## Contributing

We welcome contributions from the community to help us build and enhance Tensorlink! There are many ways to get involved:

- **Create Issues**: If you encounter bugs, have feature requests, or suggestions for improvement, please create an issue on our GitHub repository.
- **Submit Pull Requests**: Feel free to fork the repository, make changes, and submit a pull request with improvements or fixes.
- **Join the Discussion**: Reach out to us through GitHub discussions or contact us directly if you want to collaborate on specific features.

We need more people to help us refine Tensorlink and make it the best possible tool for decentralized neural network training. Your contributions and insights can make a significant impact!

## Donate

If you would like to support our work, consider buying us a coffee! Your contributions help us continue developing and improving Tensorlink.

<a href="https://www.buymeacoffee.com/smartnodes" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="width: 150px; height: auto;">
</a>
