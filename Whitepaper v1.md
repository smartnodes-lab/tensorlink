# Introduction

The prevalence of large language models across various sectors has fueled an unprecedented demand for computational power. However, current methods of model deployment, such as cloud hosting or supercomputer rentals, prove prohibitively expensive and raise concerns regarding data privacy. Furthermore, the proliferation of larger model architectures, exemplified by modern models like GPT-4 with billions of parameters, compounds accessibility issues. The sheer scale of these models demands substantial memory and computational resources for training, rendering them out of reach for many due to cost constraints. Tensorlink seeks to address these issues by offering an affordable, plug-and-play, and decentralized solution accessible to researchers and users alike. 

# Decentralized Solution for Neural Network Scaling

Tensorlink introduces a decentralized system designed to scale neural network training and inference efficiently and with the option of data obfuscation. Leveraging a network of peers, our system collectively hosts larger models, employing advanced data and model parallelism techniques to optimize training processes. By incentivizing individuals to contribute their idle compute power, we aim to build a distributed network capable of rivaling supercomputers. This approach not only democratizes access to computational resources but also offers a promising alternative to cryptocurrency mining. 

While existing approaches to distributed computation have made strides, they present limitations, such as requiring custom workflows for model distribution, being constrained by computational power, or not optimized for training. In contrast, Tensorlink aims to create a universal framework for implementing neural network offloading and acceleration in PyTorch. This plug-and-play solution is designed to automatically parse and distribute any neural network in PyTorch, including popular libraries like HuggingFace's transformers, enhancing accessibility and usability.

Additionally, Tensorlink will provide specialized workflows tailored for privacy-preserving training, ensuring that users' sensitive information remains secure throughout the neural network scaling process. This will be achieved through an innovative method of neural network submodule offloading, which allows for the obfuscation of input data and fragmentation of the original model. By leveraging this approach, Tensorlink empowers users with robust safeguards, enabling them to confidently utilize decentralized neural network scaling without compromising their privacy and security.

# P2P Architecture Overview

Validators serve as the backbone of Tensorlink’s peer-to-peer architecture, maintaining a custom distributed hash table for node connectivity, job storage, and querying. Validators help bootstrap Trainers and Worker nodes, manage active jobs by storing key training info and conducting proof of learning (PoL) checks on Workers. At the end of a job, validators aggregate information and perform multisig updates to a smart contract, adjusting worker reputation metrics, storing job proofs, and rewarding workers and validators with a native token distribution. Workers, specialized validator nodes, standby to accept distributed models or submodules from current training or inference jobs, are able to communicate with overseeing validators, and conduct inter-worker proof of learning/proof of model 

Trainers initiate job requests by bootstrapping to the network, distributing their models or forwarding them to validators with adequate memory resources. Meanwhile, Workers stand by to accept distributed models or submodules from active training or inference jobs. They maintain communication with overseeing validators and conduct inter-worker proof-of-learning or proof-of-model checks to ensure the integrity of the process.

# Distributed Models

## Pipeline Parallelism:
Pipeline parallelism is a technique used in distributed computing to overcome latency issues in neural network training. In this approach, instead of waiting for the entire batch of data to complete forward and backward passes through the model before proceeding to the next batch, the process is divided into multiple stages or "pipelines." Each pipeline handles a different stage of computation, such as forward propagation, backward propagation, and weight updates. By overlapping these stages and processing multiple batches simultaneously, we maximize hardware utilization and minimize idle time, thereby accelerating the training process.

## Data Parallelism:
To further optimize the training process, Tensorlink employs data parallelism. This involves the splitting of the training data across multiple instances of the distributed model, increasing the amount of computation done with each model pass.

## Unique Model Parsing and Distribution:
What sets Tensorlink apart is our unique approach to model parsing and distribution. Models are broken down and distributed, in a nested funcTrainers have the flexibility to maintain their custom model structures while ensuring training data privacy through obfuscation. By assigning specific submodules to workers, only a fraction of the entire model is distributed, and input data remains hidden from previous computations. This decentralized approach allows heavy computations to be offloaded while retaining control over sensitive information, as the original model resides securely on the Trainer's computer.

## Privacy Options:
For users concerned about data privacy, Tensorlink offers the option to obfuscate model and input data during distribution. Alternatively, if privacy is not a concern, users can provide the entire model and data to a validator. Validators oversee the entire training process, simplifying the workflow for users who prioritize efficiency over privacy concerns.

## Unique Model Parsing and Distribution
Tensorlink stands out due to its innovative approach to model parsing and distribution. Trainers benefit from the flexibility to maintain custom model architectures. Furthermore, users are able to host a portion of their model, which masks the input data. Likewise, model distribution can be done in such a way that prevents any individual worker from possessing your entire model. This hybrid model distribution strategy enables resource-intensive computations to be offloaded while retaining control over sensitive information.

# Proof of Learning
In the Tensorlink ecosystem, the concept of Proof of Learning (PoL) plays a key role in ensuring the integrity and reliability of distributed learning systems. Worker nodes are integral to this process, storing essential data related to model updates, input-output tensors, and gradients. Inexpensive checks between workers operating on the same subsection of a model are conducted efficiently, as parameters are already loaded. This allows for quick verification of modules, inputs/outputs, gradient updates, and other key information.

Collateralized validator nodes, on the other hand, periodically request this data throughout the training process to verify the quality of various components. Validators play a crucial role in performing more thorough and resource-intensive checks on individual workers or potentially the entire pipeline.

## Key Components of Proof of Learning:
**Gradients Validation:** Validator nodes meticulously compare the gradients stored by each worker node, ensuring consistency and correctness. Discrepancies in gradients may signal errors in the backpropagation process or differences in local datasets, prompting further investigation and correction.

**Forward Pass Validation:** Validator nodes rigorously compare the output tensors generated by each worker node during the forward pass with the expected output based on the provided input tensors. This validation step is crucial for verifying that each worker node accurately processes the data and produces reliable predictions.

**Cross-Validation:** Validator nodes leverage the data from one worker node as input to another worker node to independently validate the predictions made by each node. This cross-validation process helps identify any discrepancies or errors in the training process across different nodes, ensuring consistency and accuracy.

# Inventives

Tensorlink is powered by SmartNodes and it's native token SNO. SNO serves as the incentive for sustaining network participation while fostering the growth and development. Tokens are allocated to workers as a reward for their resources. The distribution of tokens is based on factors such as the computational power contributed, the duration of the job, and the reputation of the worker within the network. Likewise, validators are rewarded for their role in maintaining the network. To ensure their commitment and accountability, validators are required to collateralize their nodes with a specified amount of SNO tokens, thereby adding an additional layer of security and trust to the ecosystem. Lastly, SNO serves as the primary means of payment for services Tensorlink and Smartnodes.

Jobs within the Tensorlink network will commence without requiring payment, as part of the network's onboarding and incentivization strategy to attract users. However, as the network matures and scales, a transition will occur where payments for certain types of jobs will be necessitated.
