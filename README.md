# Tensorlink ![Static Badge](https://img.shields.io/badge/v0.1.0-Tensorlink-pink?logo=) ![License](https://img.shields.io/badge/License-MIT-blue.svg)


## Plug-and-Play, Peer-to-Peer Neural Network Scaling for PyTorch

**Tensorlink** is a library designed to simplify the scaling of PyTorch model training and inference, offering tools 
to easily distribute models across a network of peers and share computing resources both locally and globally. This 
approach enables the training of large models from consumer hardware, eliminating the need for cloud services for 
certain ML applications. Tensorlink leverages techniques such as automated model parsing and pipeline parallelism to 
simplify and enhance the training process, making state-of-the-art models accessible to a wider audience.

### Key features
By implementing wrappers for PyTorch's `Module` and `Optimizer` objects, Tensorlink integrates with existing codebases, 
preserving model workflows while seamlessly harnessing distributed resources. Tensorlink enables individuals and 
organizations to collaborate, share resources, and scale models dynamically—bringing the power of distributed training 
to a broader community.

* **Distributed Model Wrapper**: connects your model to a network of GPUs, managing everything from model distribution 
to execution behind the scenes
  * Supports `nn.Module` methods and queries (e.g., forward, backward, parameters)
* **Distributed Optimizer:** A coordinated optimizer that works in tandem with distributed models, supporting
essential methods like `step` and `zero_grad`
* **Node Frameworks:** Worker and Validator node frameworks for sharing computing power and securing network activities.
  * The architecture enables the creation of private networks/jobs that can function independently of the Tensorlink 
  network while benefiting from its design.

### Limitations in this Release

- Bugs, performance issues, and limited network availability are expected.
- **Model Support**: Tensorlink currently supports scriptable PyTorch models (`torch.jit.script`) and select open-source 
Hugging Face models not requiring API-keys.
   - **Why?** Security and serialization constraints in distributed workflows require deterministic models. We're 
  actively working on custom serialization methods to support all PyTorch model types. Feedback and contributions to accelerate this effort are welcome! 
- **Job Constraints**: Public jobs are currently limited to one worker due to network availability. As a result, data 
parallel acceleration is currently disabled. This will be activated as the pool of workers grows, and can also be 
enabled in the source code for local jobs/clusters.
- Internet latency and connection speeds can significantly impact performance of public jobs, which may become 
problematic for training and inference jobs. 

For a deeper dive into Tensorlink's features, capabilities, and underlying principles, please refer to the 
[Lightpaper](Lightpaper%20v1.md).

## Getting Started with Tensorlink

A more comprehensive documentation can be found on the [Tensorlink website](https://smartnodes.ca/docs)

### Installation

Before installing Tensorlink, ensure you meet the following requirements:

- UNIX/MacOS (Windows support coming soon...)
- Python 3.11.9+
- PyTorch 2.3+ (ensure model compatibility with torch version)

While we aim to reduce version requirements, Python 3.11.9+ and a UNIX-based OS are currently necessary. You can install 
Tensorlink using pip. Open your terminal and run:

```bash
pip install tensorlink
```

This command will download and install Tensorlink along with its dependencies. If you're using a virtual environment 
(recommended), ensure it's activated before running the installation command.

Tensorlink is compatible with libraries built on top of PyTorch, such as Hugging Face. Requesting a job will provide you 
with a distributed model and optimizer objects. The optimizer must be instantiated with kwargs after the request of a 
job, leaving out the .parameters() method when you call. When requesting a job, ensure that the request follows the 
instantiation of your model and precedes the training segment of your code:

```python
from tensorlink import UserNode
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Initialize tokenizer, model, optimizer, and loss function
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
loss_fn = CrossEntropyLoss()

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
Here’s an example of a training loop that distributes the model across Tensorlink nodes for accelerated computation:

```python
# Training loop
epochs = 10
for epoch in range(epochs):
    for batch in DataLoader(tokenized_dataset["train"], batch_size=8):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = distributed_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs, expected_outputs)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} completed")
```

Training progress can also be tracked through the Tensorlink/Smartnodes dashboard (TBD).

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
