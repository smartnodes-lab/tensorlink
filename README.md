# Tensorlink

## Plug-and-Play, Peer-to-Peer Neural Network Scaling for PyTorch

**Tensorlink** is a decentralized platform designed to scale neural network training and inference in PyTorch by 
distributing models across a network of peers. This innovative approach allows for efficient training and execution of 
large models on consumer hardware, providing a compelling alternative to centralized cloud services. Tensorlink 
leverages advanced techniques such as automated model parsing and pipeline parallelism to simplify and enhance the 
training process over the internet, making distributed learning accessible to a wider audience.

By implementing straightforward wrappers for `torch.nn.Module` and `torch.optim.Optimizer` objects, Tensorlink enables 
seamless integration of distributed computing without requiring any modifications to your existing codebase. This ensures 
that your workflow remains intact while you harness the power of distributed resources. With Tensorlink, users can 
effectively manage large-scale machine learning tasks, enhance model performance, and maintain data privacy.

### Key features
* **Distributed model wrapper for PyTorch**: Effortlessly connect your model to a network of GPUs, enabling you to find 
peers, parse, and distributes neural networks with ease. 
* **Seamless integration:** Handles main `nn.Module` methods and queries (e.g., forward, backward, parameters) 
in a distributed model without requiring code modifications.
* **Distributed Optimizer:** A linked optimizer that operates in conjunction with the distributed model, supporting
general methods (e.g., step, zero_grad)
* **Node Frameworks:** Worker and Validator node frameworks for donating compute and securing the network.
  * Node architecture allows for the creation of local and private training jobs separate from the Tensorlink network.


For an in-depth exploration of Tensorlink's features, capabilities, and underlying principles, please refer to the 
[Lightpaper](docs/Lightpaper v1.md).

## Getting Started with Tensorlink

**YouTube tutorial coming soon…**

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
distributed_model, distributed_optimizer = user.create_distributed_model(model, AdamW)
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

Training progress can also be tracked through the Tensorlink/Smartnodes dashboard (TBD).

### Coming Soon
* Limited worker selection and model size constraints during network launch, due to lack of workers
* In-depth model analysis/functionality beyond forwad, backward, and parameters calls to models
* Potential compatibility issues with huggingface models, Trainer objects, and optimizer steppers
* Further support for features of HuggingFace models and trainers


## Known Issues

As Tensorlink is currently in pre-alpha, our focus is on capturing and providing compute resources to users. We have chosen to release this tool to the public before implementing proof-of-learning mechanisms. **Computation may not be precisely reproducible. If you require consistent and guaranteed training results, we recommend using a different platform (BitTensor, Google Cloud, etc).** This decision was made at the protocol level, as achieving precise, reproducible distributed training across nodes requires replicating pipelines, which can be computationally expensive and inefficient. Our primary goal is to make training compute accessible to the public, but we will address this issue in the near future. 

During pre-alpha, issues may arise with gradient checkpointing, job storage, and network connectivity. Please report any problems through GitHub or our website, as your feedback is crucial to improving Tensorlink and making it more robust!

* Occasional timeouts while creating connections during swaps

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
