from src.ml.model_analyzer import handle_output
import torch.distributed.rpc as rpc
import torch
import time
import os


def run_master(splits):
    pass


def run_worker(rank, world_size, submodules):
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=128,
        rpc_timeout=10
    )

    di = torch.zeros((1, 1), dtype=torch.long)
    submodule = submodules[rank]

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

        out = handle_output(submodule(di))
        ret = rpc.rpc_sync("worker1", submodules[1], args=(out,))
        ret = rpc.rpc_sync("worker2", submodules[2], args=(handle_output(ret),))
        loss = handle_output(ret)
        loss = loss.sum()
        loss.backward()

    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

    rpc.shutdown()


def distribute_model(model, available_nodes=None, indent=0):
    class Colours:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        RESET = '\033[0m'

    if available_nodes is None:
        # Test set of nodes, real set will be obtained from either the worker's list of nodes,
        # some form of network propagation, or the smart contract
        available_nodes = [
            {"memory": 1e9, "latency_matrix": []},
            {"memory": 1e9, "latency_matrix": []},
            {"memory": 1e9, "latency_matrix": []},
            {"memory": 1e9, "latency_matrix": []}
        ]

    # While we initialize the model candidate worker nodes should be put on standby as we assign the submodules

    # Estimate memory requirements for the model
    model_memory = estimate_memory(model)
    print("   " * indent + f"Parent Module: {round(model_memory / 1e9, 3)} GB")

    # Variables for keeping track of offloaded workers + modules
    candidate_node = max(enumerate([node["memory"] for node in available_nodes]), key=lambda x: x[1])[0]
    indent += 1

    if len(list(model.children())) > 0:
        for name, submodule in model.named_children():
            submodule_memory = estimate_memory(submodule)

            # TODO:
            #  Priority: check for lowest latency x high memory node to offload first submodule to.
            #  Later: if submodule is too big we can call the distribute model again.

            # Check if best/current candidate can support the submodule
            if submodule_memory < available_nodes[candidate_node]["memory"]:
                available_nodes[candidate_node]["memory"] -= submodule_memory
                print(Colours.GREEN + "   " * indent + f"{name}: {round(submodule_memory / 1e9, 3)} GB" + Colours.RESET)
            else:
                available_nodes = available_nodes[:candidate_node] + available_nodes[candidate_node + 1:]
                candidate_node = max(enumerate([node["memory"] for node in available_nodes]), key=lambda x: x[1])[0]
                print("   " * indent + Colours.RED + "Can't accommodate sub-module on worker, distributing further..." +
                      Colours.RESET)
                distribute_model(submodule, available_nodes, indent)


def estimate_memory(module):
    """
    Dummy estimate compared to estimate_memory_requirements but doesn't require a dummy
    forward pass and thus is preferred for now.
    """
    return 4 * sum(param.numel() * param.element_size() for param in module.parameters())
