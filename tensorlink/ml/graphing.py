from tensorlink.ml.utils import estimate_memory
from transformers import PreTrainedModel
from huggingface_hub import HfApi
import torch.nn as nn
import hashlib
import random


# Create offloaded module data structure for config file
def create_offloaded(module: nn.Module, module_index: list, module_size: int):
    module_id = (
        hashlib.sha256(str(random.random()).encode()).hexdigest().encode()
    )
    data = {
        "type": "offloaded",
        "id_hash": module_id,
        "module": f"{type(module)}".split(".")[-1].split(">")[0][
                  :-1
                  ],  # class name
        "mod_id": module_index,
        "size": module_size,
        "parameters": {},
        "workers": [],
        "training": True
    }
    return module_id, data


# Create user-loaded module data structure for config file
def create_loaded(module: nn.Module, module_index: list, module_size: int):
    module_id = (
        hashlib.sha256(str(random.random()).encode()).hexdigest().encode()
    )
    data = {
        "type": "loaded",
        "id_hash": module_id,
        "module": f"{type(module)}".split(".")[-1].split(">")[0][
                  :-1
                  ],  # class name
        "mod_id": module_index,
        "size": module_size,
        "parameters": {},
        "workers": [],
    }
    return module_id, data


def find_best_worker(worker_info, module_memory):
    suitable_workers = {
        key: info for key, info in worker_info.items() if info["memory"] >= module_memory
    }

    if not suitable_workers:
        return None

    best_worker = min(suitable_workers.items(), key=lambda x: x[1]["memory"])
    return best_worker


def handle_layers(
        module: nn.Module,
        user_memory: int,
        worker_info: dict,
        config: dict = None,
        handle_layer: bool = True,
        layer_depth: int = 0,
        current_depth: int = 0,
        ids: list = None
):
    """
    Offloaded layer distribution, can modify depth of initial handling by adjusting user_memory.

    Args:
        module (nn.Module): The model or submodule to handle.
        user_memory (int): Available memory on the user's machine.
        worker_info (dict): A dictionary with workers' memory information.
        config (dict): Configuration of offloaded layers (default is None).
        handle_layer (bool): Whether the current layer should be handled before offloading.
        layer_depth (int): Minimum depth of submodules to handle before offloading (default is 0).
        current_depth (int): The current depth in the module hierarchy (default is 0).
        ids (list): A list to track the layer hierarchy (default is None).

    Returns:
        tuple: Updated configuration, remaining user memory, and worker information.
    """
    if config is None:
        config = {}
    if ids is None:
        ids = []

    # Estimate memory for the current module
    module_memory = estimate_memory(module)
    max_worker = find_best_worker(worker_info, module_memory)
    max_worker_memory = worker_info[max_worker[0]]["memory"]

    # If handle_layer is False, and we have enough memory in a worker, offload directly
    if module_memory <= max_worker_memory and handle_layer is False:
        module_id, module_info = create_offloaded(module, ids or [-1], module_memory)
        config[module_id] = module_info

    # If we have enough memory on the user's machine, load directly
    elif handle_layer and module_memory <= user_memory:
        module_id, module_info = create_loaded(module, ids or [-1], module_memory)
        config[module_id] = module_info
        user_memory -= module_memory

    else:
        # Iterate over child modules
        named_children = list(module.named_children())
        for i, (submodule_id, submodule) in enumerate(named_children):
            submodule_memory = estimate_memory(submodule)
            max_worker = find_best_worker(worker_info, submodule_memory)
            max_worker_memory = worker_info[max_worker[0]]["memory"]

            new_ids = ids + [i]

            # Decide whether to handle the layer or continue to the next depth level
            if handle_layer and user_memory >= submodule_memory and current_depth < layer_depth:
                # Handle the submodule directly if within the depth threshold
                user_memory -= submodule_memory
                submodule_id, submodule_info = create_loaded(submodule, new_ids, submodule_memory)
                config[submodule_id] = submodule_info
                handle_layer = False

            elif max_worker_memory >= submodule_memory and not handle_layer:
                # Offload to a worker if it has enough memory and initial load is not required
                submodule_id, submodule_info = create_offloaded(submodule, new_ids, submodule_memory)
                config[submodule_id] = submodule_info

            else:
                # If neither user nor worker can handle the module, further decompose it
                sub_config, user_memory, worker_info = handle_layers(
                    submodule,
                    user_memory=user_memory,
                    worker_info=worker_info.copy(),
                    config=config.copy(),
                    handle_layer=handle_layer,
                    layer_depth=layer_depth,
                    current_depth=current_depth + 1,
                    ids=new_ids
                )
                config.update(sub_config)
                handle_layer = False

    return config, user_memory, worker_info


class ModelParser:
    def __init__(self, user_memory: int, max_module_size: int = 16e9):
        self.user_memory = user_memory
        self.max_module_size = max_module_size

    def create_distributed_config(
        self,
        module: nn.Module,
        config: dict = None,
        ids: list = None,
        data_obfuscation=False
    ):
        if config is None:
            config = {}
            ids = []

        module_size = estimate_memory(module)
        module_children = list(module.named_children())

        # Offload model if under 24 Gb and data obfuscation is not required
        if data_obfuscation is False and module_size <= self.max_module_size:
            k, v = create_offloaded(module, [-1], module_size)
            config[k] = v

        # Break down module further
        else:
            for i in range(len(module_children)):
                submodule_name, submodule = module_children[i]
                submodule_memory = estimate_memory(submodule)
                submodule_type = f"{type(submodule)}".split(".")[-1].split(">")[0][:-1]

                # Update current submodule ID
                new_ids = ids + [i]

                # Handle on worker if we do not need to load on user device
                if submodule_memory <= self.max_module_size and data_obfuscation is False:
                    k, v = create_offloaded(submodule, new_ids, submodule_memory)
                    config[k] = v

                # Handle module on user device if able
                elif self.user_memory >= submodule_memory and data_obfuscation:
                    self.user_memory -= submodule_memory
                    k, v = create_loaded(submodule, new_ids, submodule_memory)
                    config[k] = v
                    data_obfuscation = False

                # Else we break down the model even further
                else:
                    sub_config = self.create_distributed_config(
                        submodule,
                        config=config.copy(),
                        ids=new_ids,
                        data_obfuscation=data_obfuscation
                    )
                    k, v = create_loaded(submodule, new_ids, submodule_memory)
                    v["subconfig"] = sub_config
                    config[k] = v

        return config
