# Tensorlink

## Plug-and-Play, Peer-to-Peer Neural Network Scaling for PyTorch

**Tensorlink** is a decentralized platform designed to scale neural network training and inference in PyTorch by distributing models across a network of peers. This enables efficient training and execution of large models on consumer hardware, offering a powerful alternative to centralized cloud services. Tensorlink leverages automated model parsing and pipeline parallelism to simplify and enhance the training process over the internet.

With a simple wrapper for `nn.Module` objects, Tensorlink allows seamless integration of distributed computing without requiring modifications to your original codebase, ensuring your workflow remains intact while harnessing distributed resources.

## Getting Started with Tensorlink

**YouTube tutorial coming soon…**

### Installation

Before installing Tensorlink, ensure you meet the following requirements:

- UNIX/MacOS (Windows support coming soon...)
- Python 3.11.9+
- PyTorch 2.3+ (ensure model compatibility with torch version)

While we aim to reduce version requirements, Python 3.11.9+ and a UNIX-based OS are currently necessary. You can install Tensorlink using pip. Open your terminal and run:

```bash
pip install tensorlink
```

This command will download and install Tensorlink along with its dependencies. If you're using a virtual environment (recommended), ensure it's activated before running the installation command.

Tensorlink is compatible with libraries built on top of PyTorch, such as Hugging Face. When leveraging Tensorlink, ensure that the distributed request follows the instantiation of your model and precedes the training segment of your code:

```python
from tensorlink import UserNode
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Initialize tokenizer, model, optimizer, and loss function
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

# Create a user node and connect to Tensorlink
user = UserNode()

# Create a distributed model and connect to Tensorlink
distributed_model = user.create_distributed_model(model, optimizer)
```

Once the job request is created, you'll be successfully connected to Tensorlink. You can now proceed with training. Here’s an example of a training loop that distributes the model across Tensorlink nodes for accelerated computation:

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
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} completed")
```

You can monitor the progress of your distributed training job using Tensorlink’s job status functionality:

```python
# Check job status
status = user.check_job_status()
print(f"Job Status: {status}")
```

Training progress can also be tracked through the Tensorlink/Smartnodes dashboard.

## Known Issues

**Computation is not guaranteed to be Turing complete. If you require precise, replicable training, we advise using a different platform (BitTensor, Google Cloud, etc)!** This is a protocol-level decision, as Turing-complete neural network computation can be expensive and inefficient. Our primary goal is to provide training compute for the public. As Tensorlink is currently in pre-alpha, we focus on capturing and providing compute resources to users. We have opted to release this tool to the public before enforcing proof-of-learning.

During pre-alpha, issues may arise with gradient checkpointing, job storage, and network connectivity. Please report any problems through GitHub or our website, as your feedback is crucial to improving Tensorlink and making it more robust!


## Contributing

We welcome contributions from the community to help us build and enhance Tensorlink! There are many ways to get involved:

- **Create Issues**: If you encounter bugs, have feature requests, or suggestions for improvement, please create an issue on our GitHub repository.
- **Submit Pull Requests**: Feel free to fork the repository, make changes, and submit a pull request with improvements or fixes.
- **Join the Discussion**: Reach out to us through GitHub discussions or contact us directly if you want to collaborate on specific features.

We need more people to help us refine Tensorlink and make it the best possible tool for decentralized neural network training. Your contributions and insights can make a significant impact!

## Donate

If you would like to support our work, consider buying us a coffee! Your contributions help us continue developing and improving Tensorlink.

[![Buy Me a Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://www.buymeacoffee.com/smartnodes)