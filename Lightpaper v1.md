## Introduction

The prevalence of large language models has fueled an unprecedented demand for computational power. Modern models, like 
GPT-4 with hundreds of billions of parameters, necessitate vast arrays of graphics or tensor processing units to train. 
Traditional deployment methods, such as cloud services or supercomputer rentals, while effective, often come 
with prohibitive costs. Furthermore, the setup of distributed models, whether on a local cluster or offloaded over the 
internet, requires extensive customization of workflows to distribute and coordinate modules across nodes. Tensorlink 
aims to address these challenges by offering a universal, plug-and-play framework for model offloading and acceleration 
in PyTorch, automatically parsing and distributing any model to reduce complexity and improve accessibility. 


## A Decentralized Solution for Neural Network Scaling

Tensorlink introduces a versatile framework designed to facilitate neural network offloading and acceleration within 
PyTorch, a leading machine learning framework in Python. Tensorlink offers a straightforward, plug-and-play solution 
that seamlessly parses and distributes any PyTorch-based neural network model, including transformers utilized in 
libraries such as HuggingFace. Through this approach, Tensorlink aims to overcome existing economical and technical 
barriers by offering an affordable, plug-and-play, decentralized solution that enhances accessibility and ease of use 
for researchers, businesses, and individuals alike.

Users are provided with a model wrapper that handles model distribution and connects them directly to a network of 
computing power, while maintaining their existing workflows on their own device. This wrapper gives the perception of 
operating models on a user's own hardware, which reduces complexity, especially for larger models that require hosting
across multiple devices. By incentivizing the contribution of idle computing power through a smart contract-based 
payments and rewards (via Smartnodes), Tensorlink aspires to cultivate a distributed network that competes with the 
capabilities of supercomputers. This strategy will democratize access to computational resources and serve as a viable 
alternative to traditional cryptocurrency mining.


### Scaling Models Over the Internet

In order to achieve the efficient scaling of neural networks over the internet, we must leverage certain training 
parallelism techniques.

* **Data parallelism:** a technique that involves splitting the input data into smaller subsets that can be processed 
simultaneously across multiple devices. Each device receives a copy of the entire model and processes a different subset 
of the data. Once all devices have computed their results for their respective data shards, the gradients from each node
are averaged and synchronized to update the model parameters. This approach is widely used in large-scale training 
because it is straightforward to implement and scales well across multiple machines.


* **Model parallelism:** divides a neural network model itself across multiple devices, with each device holding only a 
portion of the model layers. When data is processed, it moves sequentially through each device as it progresses through 
different model segments. This approach is beneficial when models are too large to fit entirely on a single device. For 
example, with a very large neural network, the first few layers might be on one GPU, the next layers on another, and so 
forth.


* **Pipeline parallelism:**  a hybrid approach that combines aspects of model parallelism to reduce latency by dividing 
the model across devices but also overlaps the execution of multiple data batches. Instead of processing one data batch 
through all model layers sequentially, the model is split into stages or "pipelines," each handling a portion of the 
model’s layers. Data batches are fed into the pipeline in a staggered fashion, so while one batch is being processed by 
one stage, the next batch can start processing in a different stage. This overlapping reduces idle time and can 
significantly speed up training by maximizing throughput.

<p align="center">
  <img src="docs/pipeline.png" alt="Distributed model architecture." width="520"/>
</p>
<p align="center"><strong>Figure 1:</strong> <em>Pipeline parallelism micro-batching to mitigate latency in distributed learning environments.</em></p>


### Model Parsing and Distribution

Tensorlink provides algorithms for analyzing and parsing neural networks in the context of distributed training. These 
algorithms take factors such as memory usage, the number of workers/intermediates required in the workflow data, and 
available network resources, to optimize the distribution of models across the network.

For users comfortable sharing their data, Tensorlink allows the distribution of full models and datasets to validators. 
This method minimizes the impact of internet latency, as it involves a limited number of offloaded modules in the 
pipeline. Additionally, models can be distributed in a privacy-oriented approach to model parsing and distribution for 
those prioritizing data security. By further fragmenting model structures, only a small portion of the model is 
accessible to any single node. Users can further safeguard their training data by employing hybrid distributed models,
which the processing of a small portion of the model on the user side before the data continues through the pipeline. 
This effectively obfuscates sensitive model and training data, ensuring a more secure training environment, potentially 
at the potential sacrifice of additional training time or computational cost.


<p align="center">
  <img src="docs/ML Flow Chart.png" alt="Distributed model architecture." width="520"/>
</p>
<p align="center"><strong>Figure 2:</strong> <em>An example illustrating model distribution and parallelization among
workers, as well as intermediate computation by the user.</em></p>


### Proof of Learning
***Currently in development***

Proof of Learning (PoL) is a foundational concept of Tensorlink, designed to enhance the integrity and reliability 
of distributed learning. During the training processes, worker nodes are responsible for storing critical data related 
to model updates, input-output tensors, and gradients. Validators are tasked with periodically requesting this data 
throughout the training process to verify its validity and ensure consistency among worker nodes.

In Tensorlink’s PoL process, efficient, low-cost checks occur within workers handling replica model sections, allowing 
for rapid validation among works without incurring a significant computational costs. For more comprehensive verification,
collateralized validators perform in-depth checks across individual workers and larger model sections to ensure 
reliability and accuracy of training.

* **Gradients Validation:** Validator nodes compare stored gradient values across worker nodes to ensure alignment and 
accuracy. Any gradient discrepancies may indicate potential malicious activity, signaling the need for further 
inspection.
  
* **Forward Pass Validation:** Validators assess the output tensors generated during forward passes by comparing them to
expected outputs based on provided input tensors. This step is essential to ensure each worker node processes data 
accurately and produces reliable predictions.

* **Cross-Validation:** Validator nodes cross-check data between worker nodes by taking data from one node and using it 
as input in another. This independent validation ensures consistency across nodes, helping to identify any discrepancies 
that might impact model accuracy.


## Incentives

Tensorlink is powered by Smartnodes and it's native token, SNO. SNO functions as a vital incentive for encouraging 
network participation while promoting growth and development. Tokens are allocated to workers as rewards for their 
contributions of computational resources, with distribution determined by factors such as the computational power 
provided, job duration, and the worker's reputation within the network. Validators also receive rewards for their 
essential role in maintaining the network's integrity. To ensure their commitment and accountability, validators must 
collateralize their nodes with a specified amount of SNO tokens, adding an extra layer of security and trust to the 
ecosystem. Furthermore, SNO serves as the primary means of payment for services Tensorlink and Smartnodes.

To foster user engagement and research, jobs within the Tensorlink network will initially commence without requiring 
payment as part of an onboarding and incentivization strategy to attract users. However, as the network matures and 
scales, a gradual transition will occur, necessitating payments for certain types of jobs to sustain network operation.
