# Introduction

The prevalence of large language models across various sectors has fueled an unprecedented demand for computational power. The sheer scale of these models demands substantial memory and computational resources for training, rendering them out of reach for many. Current methods of model deployment, such as cloud hosting or supercomputer rentals are either expensive, require custom distributed implementation, or raise concerns regarding data and model privacy. TensorLink seeks to address these issues by offering an affordable and decentralized solution accessible to researchers and users alike. 

# Decentralized Solution for Neural Network Scaling

TensorLink introduces a decentralized system designed to scale neural network training and inference efficiently. Leveraging a network of peers, our system collectively hosts larger models, employing advanced data and model parallelism techniques to optimize training processes. By incentivizing individuals to contribute their idle compute power, we aim to build a distributed network capable of rivaling supercomputers. This approach not only democratizes access to computational resources but also offers a promising alternative to cryptocurrency mining. 

While existing approaches to distributed computation, such as BOINC, Akash, and Flux, have made strides, they present limitations in machine-learning specific tasks, requiring custom implementation for model distribution or being constrained by the computational power of individual nodes. In contrast, TensorLink aims to create a universal framework for implementing neural network offloading and acceleration in PyTorch. This plug-and-play solution seamlessly parses and distributes any neural network in PyTorch, including models from popular libraries like HuggingFace, enhancing accessibility and usability of these larger models.

Moreover, TensorLink offers specialized workflows designed for privacy-preserving training, safeguarding users' sensitive information throughout the neural network scaling process. This will be realized through an innovative approach to neural network submodule offloading, enabling the obfuscation of input data and the fragmentation of models. While exploring additional privacy methods like homomorphic encryption holds promise for the future, significant technological breakthroughs or advancements in research are required to fully implement them in this context.

# P2P Architecture Overview

In TensorLink's peer-to-peer (P2P) architecture, Validators play a crucial role, serving as the backbone for connectivity, job management, and verification within the network. They maintain a custom distributed hash table, facilitating node connectivity and storing essential job information.

## Validators
Validators bootstrap Trainers and Worker nodes, ensuring seamless integration into the network. They manage active jobs by storing key training data and conducting proof of learning (PoL) checks on Workers. Upon completion of a job, Validators aggregate information and execute multisig updates to a smart contract. This involves adjusting worker reputation metrics, storing job proofs, and distributing rewards to workers and validators, which may include payments or token generation.

## Workers
Worker nodes, which function as specialized validator nodes, are primed to accept training or inference jobs from Validators. They establish connections with Trainers and other workers, both before and after their role in the model. During training, parallel workers with the same submodule loaded collaborate for gradient averaging. Moreover, workers maintain communication channels with overseeing validators and facilitate proof of learning/proof of model requests between validators and parallel workers.

## Pipeline Parallelism

**Pipeline parallelism** is a powerful technique utilized in distributed computing to mitigate latency issues during neural network training. This methodology involves breaking down the training process into multiple stages or "pipelines," where each pipeline handles a distinct computation task. By concurrently executing these stages and processing multiple micro-batches in parallel, pipeline parallelism effectively minimizes the overall training duration in a distributed environment.

## Unique Model Parsing and Distribution
TensorLink stands out due to its innovative approach to model parsing and distribution. Trainers benefit from the flexibility to maintain custom model architectures. The generalized model wrapper and distribution process means this framework can be used across a large array of PyTorch neural networks. 

## Privacy Options:
By assigning specific submodules to workers, only a portion of the complete model is distributed, ensuring that input data remains concealed from prior computations. This decentralized model distribution strategy enables resource-intensive computations to be offloaded while retaining control over sensitive information, as the original model remains securely stored on the Trainer's system. For users concerned about data privacy, TensorLink offers the option to obfuscate model and input data during distribution. Alternatively, if privacy is not a concern, users can provide the entire model and data to a validator. Validators oversee the entire training process, simplifying the workflow for users who prioritize efficiency over privacy concerns.


# Proof of Learning

In the TensorLink ecosystem, the concept of Proof of Learning (PoL) plays a pivotal role in ensuring the integrity and reliability of distributed learning systems. Worker nodes are integral to this process, storing essential data related to model updates, input-output tensors, and gradients. Validator nodes, on the other hand, periodically request this data throughout the training process to verify the validity of various components, thereby enhancing transparency and accountability. 

Inexpensive checks between workers operating on the same subsection of the model are conducted efficiently, as modules are already loaded, allowing for quick verification of proof of model, input/output, gradient updates, and other parameters. Collateralized validators play a crucial role in performing more thorough and resource-intensive checks on individual workers or potentially the entire pipeline, ensuring robustness and reliability in the learning process.


## Key Components of Proof of Learning:

Gradients Validation: Validator nodes meticulously compare the gradients stored by each worker node, ensuring consistency and correctness. Discrepancies in gradients may signal errors in the backpropagation process or differences in local datasets, prompting further investigation and correction.

Forward Pass Validation: Validator nodes rigorously compare the output tensors generated by each worker node during the forward pass with the expected output based on the provided input tensors. This validation step is crucial for verifying that each worker node accurately processes the data and produces reliable predictions.

Cross-Validation: Validator nodes leverage the data from one worker node as input to another worker node to independently validate the predictions made by each node. This cross-validation process helps identify any discrepancies or errors in the training process across different nodes, ensuring consistency and accuracy.

# Tokenomics

TensorLink's token ecosystem, powered by SmartNodes token (SNO), serves as the cornerstone of incentivizing and sustaining network participation while fostering the growth and development of the decentralized learning ecosystem.

Rewarding Network Participation: SNO tokens are allocated as rewards to workers upon the successful completion of a job. The distribution of tokens is determined based on factors such as the computational power contributed, the duration of the job, and the reputation of the worker within the network.

Collateralization of Nodes: Validators play a crucial role in maintaining the integrity and security of the network. To ensure their commitment and accountability, validators are required to collateralize their nodes with a specified amount of SNO tokens, thereby adding an additional layer of security and trust to the ecosystem.

Payments for Services: SNO tokens serve as the primary means of payment for services rendered within the TensorLink network. Users can utilize SNO tokens to pay for various services, including accessing computational resources, deploying models, and utilizing privacy-preserving features.

Tokens are distributed dynamically, reflecting the contributions and efforts of participants within the network. Workers receive their share of tokens proportional to their contributions, incentivizing active participation and engagement. Similarly, validators earn a percentage of rewards for their role in moderating and maintaining the network's integrity.

Initially, jobs within the TensorLink network will commence without requiring payment, as part of the network's onboarding and incentivization strategy to attract users. However, as the network matures and scales, a transition will occur where payments for certain types of jobs, such as those involving very large models, will be necessitated. These payments will be facilitated using SNO, ensuring the sustainability and growth of the ecosystem.