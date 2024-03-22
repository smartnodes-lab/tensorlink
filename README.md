# TensorLink

TensorLink is a decentralized platform designed to scale neural network training and inference in PyTorch by arbitrarily distributing models across a network of peers. With TensorLink, users can efficiently train and execute large models, harnessing the combined computational power of distributed nodes.

## Key Features:

- **Model Distribution:** TensorLink leverages pipeline parallelism, distributing model parameters across multiple nodes, allowing for accelerated training and larger models. This feature enables users to scale the models they can train on their own devices while accelerating the training process, addressing two significant challenges in machine learning utilization.
  
- **Data Privacy and Security:** TensorLink provides workflows for obfuscating input data and fragmenting model structures, ensuring the privacy of sensitive information. Users can securely leverage the platform without compromising data confidentiality.

- **Consensus:** TensorLink implements a novel proof-of-learning consensus mechanism to ensure the integrity and accuracy of distributed learning processes. Collateralized Validators play a crucial role in maintaining the network's reliability and transparency as well.

## TODO: Getting Started / Examples

1. **Installation:** Install TensorLink using pip: `pip install tensorlink`.

2. **Bootstrap to TensorLink:** Learn how to join the network as a worker and/or validator node to contribute compute power and earn rewards.

3. **Running Local Training Network:** Follow provided examples to set up a simple distributed training scenario using TensorLink on a local computer or cluster.

4. **Deploying a Job:** Experiment with distributed model parameters for various offloading techniques such as training, inference, and privacy-preserved training.

5. **Contributing to Development:** Get involved in the TensorLink open-source community by contributing code, reporting issues, or providing feedback on the project's development roadmap.
