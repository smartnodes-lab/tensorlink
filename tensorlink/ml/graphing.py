import hashlib
import random

import torch.nn as nn
from transformers import PreTrainedModel
from typing import Union

from tensorlink.ml.utils import estimate_memory, estimate_hf_model_memory


# Create offloaded module data structure for config file
def create_offloaded(
    module: Union[nn.Module, str], module_index: list, module_size: int
):
    module_id = hashlib.sha256(str(random.random()).encode()).hexdigest()
    data = {
        "type": "offloaded",
        "id_hash": module_id,
        "module": (
            f"{type(module)}".split(".")[-1].split(">")[0][:-1]
            if not isinstance(module, str)
            else module
        ),
        "mod_id": module_index,
        "size": module_size,
        "workers": [],
        "training": True,
    }
    return module_id, data


# Create user-loaded module data structure for config file
def create_loaded(module: nn.Module, module_index: list, module_size: int):
    module_id = hashlib.sha256(str(random.random()).encode()).hexdigest()
    data = {
        "type": "loaded",
        "id_hash": module_id,
        "module": f"{type(module)}".split(".")[-1].split(">")[0][:-1],
        "mod_id": module_index,
        "size": module_size,
        "workers": [],
    }
    return module_id, data


def find_best_worker(worker_info, module_memory):
    suitable_workers = {
        key: info
        for key, info in worker_info.items()
        if info["memory"] >= module_memory
    }

    if not suitable_workers:
        return None

    best_worker = min(suitable_workers.items(), key=lambda x: x[1]["memory"])
    return best_worker


def handle_layer(
    module: nn.Module,
    user_memory: int,
    worker_info: dict,
    config: dict = None,
    handle_layers: bool = True,
    layer_depth: int = 0,
    current_depth: int = 0,
    ids: list = None,
):
    if config is None:
        config = {}
    if ids is None:
        ids = []

    # Estimate memory for the current module
    module_memory = estimate_memory(module)
    max_worker = find_best_worker(worker_info, module_memory)
    max_worker_memory = worker_info[max_worker[0]]["memory"]

    # If handle_layer is False, and we have enough memory in a worker, offload directly
    if module_memory <= max_worker_memory and handle_layers is False:
        module_id, module_info = create_offloaded(module, ids or [-1], module_memory)
        config[module_id] = module_info

    # If we have enough memory on the user's machine, load directly
    elif handle_layers and module_memory <= user_memory:
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
            if (
                handle_layers
                and user_memory >= submodule_memory
                and current_depth < layer_depth
            ):
                # Handle the submodule directly if within the depth threshold
                user_memory -= submodule_memory
                submodule_id, submodule_info = create_loaded(
                    submodule, new_ids, submodule_memory
                )
                config[submodule_id] = submodule_info
                handle_layers = False

            elif max_worker_memory >= submodule_memory and not handle_layers:
                # Offload to a worker if it has enough memory and initial load is not required
                submodule_id, submodule_info = create_offloaded(
                    submodule, new_ids, submodule_memory
                )
                config[submodule_id] = submodule_info

            else:
                # If neither user nor worker can handle the module, further decompose it
                sub_config, user_memory, worker_info = handle_layer(
                    submodule,
                    user_memory=user_memory,
                    worker_info=worker_info.copy(),
                    config=config.copy(),
                    handle_layers=handle_layers,
                    layer_depth=layer_depth,
                    current_depth=current_depth + 1,
                    ids=new_ids,
                )
                config.update(sub_config)
                handle_layers = False

    return config, user_memory, worker_info


class ModelParser:
    def __init__(self, user_memory: int = 0):
        self.user_memory = user_memory
        self.gpu_sizes = [1e9, 2e9, 4e9, 8e9, 12e9, 16e9, 24e9]

    def create_distributed_config(
        self,
        model: nn.Module,
        training: bool,
        trusted: bool,
        handle_layers: bool = True,
        input_obfuscation: bool = False,
    ):
        """
        Creates a distributed configuration for a model, determining how it should be allocated across nodes.

        :param model: The neural network model to distribute.
        :param training: Whether the model will be trained.
        :param trusted: Whether the model is trusted (determines naming policies).
        :param handle_layers: Whether to process individual layers for fine-grained distribution.
        :param input_obfuscation: If True, prioritizes local execution for privacy.
        :return: A dictionary representing the distributed model configuration.
        """
        return self._recurse_module(
            model, training, trusted, handle_layers, input_obfuscation
        )

    def _recurse_module(
        self,
        module: Union[nn.Module, str],
        training: bool,
        trusted: bool,
        handle_layers: bool,
        input_obfuscation: bool,
        config: dict = None,
        ids: list = None,
    ):
        if config is None:
            config, ids = {}, []

        # Estimate memory usage
        if isinstance(module, str):
            module_size, _ = estimate_hf_model_memory(module, training=training)
            module_name = module
        else:
            module_size = estimate_memory(module, training, batch_size=1024)
            module_name = getattr(module, 'name_or_path', str(type(module)))

        # Select the smallest GPU size that is still > module_size
        try:
            min_gpu_size = min([s for s in self.gpu_sizes if s >= module_size])
        except ValueError:
            print("❌ Model too large to fit in any available GPU.")
            print(f"📦 Module: {module_name}")
            print(f"🧠 Estimated Size: {module_size:.2f} GB")
            print(f"🖥️ Available GPU Sizes: {[f'{s:.2f} GB' for s in self.gpu_sizes]}")
            raise RuntimeError(
                "🚫 This model is too large to fit on any of the available GPUs.\n"
                "Currently, public jobs only support HuggingFace models with memory usage within device limits.\n"
                "Model: {}\n"
                "Estimated Size: {:.2f} GB\n"
                "Smallest Available GPU: {:.2f} GB\n".format(
                    module_name, module_size, min(self.gpu_sizes)
                )
            )

        # Assign module to config
        k, v = create_offloaded(module, [-1], module_size)
        v["name"] = None

        if isinstance(module, PreTrainedModel):
            v["name"] = module.name_or_path
        elif isinstance(module, str):
            v["name"] = module
        elif not trusted:
            v["name"] = module.name_or_path

        config[k] = v

        # if input_obfuscation is True and not handled_layer and module_size <= self.user_memory:
        #     k, v = create_loaded(module, ids)
        #
        # # Offload model if under 24 Gb and data obfuscation is not required
        # elif input_obfuscation is False and module_size <= min_gpu_size:
        #     k, v = create_offloaded(module, [-1], module_size)
        #     config[k] = v
        #
        # # Break down module further
        # else:
        #     for i in range(len(module_children)):
        #         submodule_name, submodule = module_children[i]
        #         submodule_memory = estimate_memory(submodule)
        #         submodule_type = f"{type(submodule)}".split(".")[-1].split(">")[0][:-1]
        #
        #         # Update current submodule ID
        #         new_ids = ids + [i]
        #
        #         # Handle on worker if we do not need to load on user device
        #         if (
        #             submodule_memory <= self.max_module_size
        #             and data_obfuscation is False
        #         ):
        #             k, v = create_offloaded(submodule, new_ids, submodule_memory)
        #             config[k] = v
        #
        #         # Handle module on user device if able
        #         elif self.user_memory >= submodule_memory and data_obfuscation:
        #             self.user_memory -= submodule_memory
        #             k, v = create_loaded(submodule, new_ids, submodule_memory)
        #             config[k] = v
        #             data_obfuscation = Falsei8
        #
        #         # Else we break down the model even further
        #         else:
        #             sub_config = self.create_distributed_config(
        #                 submodule,
        #                 config=config.copy(),
        #                 ids=new_ids,
        #                 data_obfuscation=data_obfuscation,
        #             )
        #             k, v = create_loaded(submodule, new_ids, submodule_memory)
        #             v["subconfig"] = sub_config
        #             config[k] = v

        return config
