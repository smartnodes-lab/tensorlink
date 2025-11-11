from tensorlink.ml.utils import estimate_memory, estimate_hf_model_memory

from transformers import PreTrainedModel, AutoModel, AutoConfig
from accelerate import init_empty_weights
from collections import defaultdict
from typing import Union, Optional
import torch.nn as nn
import textwrap
import hashlib
import inspect
import random


class AssignmentError(Exception):
    """Raised when a module cannot be assigned to any worker."""

    pass


class ModelParser:
    def __init__(self, user_memory: int = 0):
        self.user_memory = user_memory
        self.model_name = ""
        self.assigned_workers = defaultdict(list)
        self.forward_code_cache = {}
        self.verbose = True

    def create_distributed_config(
        self,
        model: Union[nn.Module, str],
        workers: dict,
        training: bool,
        trusted: bool,
        handle_layers: bool = True,
        input_obfuscation: bool = False,
        optimizer_type: str = "adam",
    ):
        """
        Creates a distributed configuration for a model, determining how it should be allocated across nodes.

        :param model: The neural network model to distribute.
        :param workers: dict of available worker candidates to offload models to.
        :param training: Whether the model will be trained.
        :param trusted: Whether the model is trusted (determines naming policies).
        :param handle_layers: Whether to process individual layers for fine-grained distribution.
        :param input_obfuscation: If True, prioritizes local execution for privacy.
        :param optimizer_type: Name of the optimizer type.
        :return: A dictionary representing the distributed model configuration.
        """

        if isinstance(model, str):
            self.model_name = model
            model_config = AutoConfig.from_pretrained(model)
            with init_empty_weights():
                model = AutoModel.from_config(model_config)

        workers_state = {
            wid: {"gpu_memory": w["gpu_memory"], "original_memory": w["gpu_memory"]}
            for wid, w in workers.items()
        }

        config = {}
        success = True

        try:
            config, _ = self._recurse_module(
                module=model,
                parent_module=None,
                module_path="model",
                workers_state=workers_state,
                training=training,
                trusted=trusted,
                handle_layers=handle_layers,
                input_obfuscation=input_obfuscation,
                last_worker=None,
                optimizer_type=optimizer_type,
            )

        except AssignmentError:
            success = False

        return {"success": success, "config": config}

    def _recurse_module(
        self,
        module: nn.Module,
        parent_module: Optional[nn.Module],
        module_path: str,
        workers_state: dict,
        training: bool,
        trusted: bool,
        handle_layers: bool,
        input_obfuscation: bool,
        last_worker: Optional[str] = None,
        depth: int = 0,
        ids: list = None,
        optimizer_type="adam",
    ):
        config = {}
        if ids is None:
            ids = []

        memory, breakdown = estimate_memory(
            module, training, batch_size=1024, optimizer_type=optimizer_type
        )

        assigned_worker = self._try_assign_worker(
            memory, module_path, workers_state, last_worker
        )

        # full module fits on a worker
        if assigned_worker:
            config[module_path] = {
                "type": "offloaded",
                "name": self.model_name,
                "assigned_workers": [assigned_worker],
                "module_id": ids,
                "memory": memory,
                "module": (
                    f"{type(module)}".split(".")[-1].split(">")[0][:-1]
                    if not isinstance(module, str)
                    else module
                ),
                "training": training,
                "optimizer_type": optimizer_type,
            }

            self.assigned_workers[assigned_worker].append(
                {"module_id": ids, "memory": memory, "module": module}
            )

            return config, assigned_worker

        # too large, recurse into children
        if self.verbose:
            print(
                f"Module {module_path} ({memory / 1e9:.2f}GB) too large, recursing into children..."
            )

        children = list(module.named_children())
        if not children:
            config[module_path] = {"type": "unassigned", "required_memory": memory}
            raise AssignmentError(f"Unable to assign {module_path}")

        parent_forward_code = self._extract_forward_code(module)
        child_workers = set()
        prev_child_worker = last_worker
        last_successful_worker = last_worker

        for child_name, child_module in children:
            child_path = f"{module_path}.{child_name}"

            try:
                child_config, child_last_worker = self._recurse_module(
                    module=child_module,
                    parent_module=module,
                    module_path=child_path,
                    workers_state=workers_state,
                    training=training,
                    trusted=trusted,
                    last_worker=prev_child_worker,
                    handle_layers=handle_layers,
                    input_obfuscation=input_obfuscation,
                    depth=depth + 1,
                    optimizer_type=optimizer_type,
                )

                config.update(child_config)
                if child_last_worker:
                    prev_child_worker = child_last_worker
                    last_successful_worker = child_last_worker
                    child_workers.add(child_last_worker)

            except AssignmentError:
                # if recursion fails, we keep the last successful worker
                prev_child_worker = last_successful_worker

        if len(child_workers) > 1 and parent_forward_code:
            for child_path, child_cfg in config.items():
                if child_cfg.get("assigned_workers", [None])[0] in child_workers:
                    child_cfg["parent_forward_code"] = parent_forward_code
                    child_cfg["parent_module_path"] = module_path

        return config, last_successful_worker

    def _try_assign_worker(
        self,
        memory: float,
        module_path: str,
        workers_state: dict,
        last_worker: Optional[str],
    ):
        """
        Try to assign module to a worker, preferring last worker
        """
        # Sort workers, using the previous worker first, and then from descending capacity
        worker_priority = []
        for wid, winfo in workers_state.items():
            if wid == last_worker:
                worker_priority.insert(0, (wid, winfo))
            else:
                worker_priority.append((wid, winfo))

        if len(worker_priority) > 1:
            first_worker = worker_priority[0]
            rest = sorted(
                worker_priority[1:], key=lambda x: x[1]["gpu_memory"], reverse=True
            )
            worker_priority = [first_worker] + rest

        # Try to assign to a worker
        for worker_id, worker_info in worker_priority:
            if worker_info["gpu_memory"] >= memory:
                worker_info["gpu_memory"] -= memory
                if self.verbose:
                    print(f"Assigned {module_path} ({memory/1e9:.2f}GB) to {worker_id}")
                return worker_id

        return None

    def _extract_forward_code(self, module: nn.Module):
        """
        Extract the forward pass logic from the source code. Allows workers to
        execute the parent's forward logic locally.
        """
        # Check cache
        module_class = type(module)
        if module_class in self.forward_code_cache:
            return self.forward_code_cache[module_class]

        try:
            forward_method = module.forward
            source = inspect.getsource(forward_method)
            source = textwrap.dedent(source)
            self.forward_code_cache[module_class] = source
            return source

        except (OSError, TypeError) as e:
            if self.verbose:
                print(
                    f"Could not extract forward code for {module_class.__name__}: {e}"
                )
            return None
