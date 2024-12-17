from tensorlink.mpc.shared_memory import get_from_shared_memory, store_in_shared_memory
from tensorlink.ml.utils import *
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoModelForNextSentencePrediction,
    AutoModelForMultipleChoice,
    AutoModelForPreTraining,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForSemanticSegmentation,
    AutoModelForObjectDetection,
    AutoModelForAudioClassification,
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoModelForVision2Seq,
)
from huggingface_hub import HfApi, hf_hub_download
from collections import deque
import threading
import logging
import pickle
import torch
import queue
import json
import time
import os


base_dir = os.path.dirname(os.path.abspath(__file__))


MODEL_TYPE_MAPPING = {
    'ForSequenceClassification': AutoModelForSequenceClassification,
    'ForTokenClassification': AutoModelForTokenClassification,
    'ForQuestionAnswering': AutoModelForQuestionAnswering,
    'ForMaskedLM': AutoModelForMaskedLM,
    'ForNextSentencePrediction': AutoModelForNextSentencePrediction,
    'ForMultipleChoice': AutoModelForMultipleChoice,
    'ForPreTraining': AutoModelForPreTraining,
    'ForCausalLM': AutoModelForCausalLM,
    'ForImageClassification': AutoModelForImageClassification,
    'ForSemanticSegmentation': AutoModelForSemanticSegmentation,
    'ForObjectDetection': AutoModelForObjectDetection,
    'ForAudioClassification': AutoModelForAudioClassification,
    'ForCTC': AutoModelForCTC,
    'ForSpeechSeq2Seq': AutoModelForSpeechSeq2Seq,
    'ForVision2Seq': AutoModelForVision2Seq,
}


class DistributedWorker:
    def __init__(self, node_requests, node_responses, mpc_lock):
        self.node_requests = node_requests
        self.node_responses = node_responses
        self.mpc_lock = mpc_lock
        self.rolling_buffer = deque(maxlen=10)
        self.storage_path = "./tmp/snapshots"

        self.modules = {}
        self.optimizers = {}
        self.terminate = False
        self.lock = threading.Lock()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_loop(self):
        while not self.terminate:
            for module_id in list(self.modules.keys()):
                module = self.modules.get(module_id)

                # Handle backward pass
                if not module.backward_queue.empty():
                    n_batch = module.n_batch
                    next_node = module.host

                    if self.modules[module_id].training:
                        # Critical section: lock the shared resources only when necessary
                        with self.lock:
                            tag, loss_relay = module.backward_queue.get()
                            tensor_bytes = get_from_shared_memory(loss_relay[0], loss_relay[1], encoded=True)
                            tensor = bytes_to_tensor(tensor_bytes)
                            # Load loss and move to the device
                            loss = attach_tensor(
                                tensor,
                                self.device
                            )

                            inter_tag = tuple(tag)
                            assoc_input, assoc_output = module.intermediates.pop(inter_tag)

                            assoc_input = assoc_input.to(self.device)
                            assoc_output = assoc_output.to(self.device)

                            # Backward pass with CUDA synchronization for accurate profiling
                            if self.device.type != "cpu":
                                torch.cuda.synchronize()

                            assoc_output.backward(loss)

                            if self.device.type != "cpu":
                                torch.cuda.synchronize()

                            # Detach gradients and prepare for next node
                            if assoc_input.grad is None:
                                dvalues = detach_tensor(torch.zeros_like(assoc_input, dtype=torch.float32))
                            else:
                                dvalues = detach_tensor(assoc_input.grad)

                            # Clean up to avoid memory leaks
                            del assoc_input, assoc_output

                            # Store pass in shared memory and send to next node
                            dvalues_bytes = json.dumps(tensor_to_bytes(dvalues)).encode()
                            size, name = store_in_shared_memory(dvalues_bytes, encoded=True)
                            self.send_request("send_backward", (next_node, size, name, tag))

                            # Clear memory, but avoid excessive cache clearing
                            if self.device.type != "cpu":
                                torch.cuda.empty_cache()

                # Handle forward pass
                if not module.forward_queue.empty():
                    with self.lock:
                        key, (size, name) = module.forward_queue.get()

                        tensor_bytes = get_from_shared_memory(size, name, encoded=True)
                        args, kwargs = tensor_bytes.split(b"|")
                        args = bytes_to_tensor(args)
                        kwargs = bytes_to_tensor(kwargs)

                        # Enable gradient tracking and move to device
                        inp = enable_grad(attach_tensor(args, self.device))
                        kwargs = enable_grad(attach_tensor(kwargs, self.device))

                        # Forward pass with CUDA synchronization for accurate profiling
                        if self.device.type != "cpu":
                            torch.cuda.synchronize()

                        out = module(inp, **kwargs)

                        if self.device.type != "cpu":
                            torch.cuda.synchronize()

                        # Store intermediate results if training
                        if self.modules[module_id].training:
                            module.intermediates[key] = [inp, handle_output(out).to(self.device)]

                        # Detach and store output
                        detached_out = detach_tensor(out)
                        output_bytes = json.dumps(
                            tensor_to_bytes(detached_out)
                        ).encode()
                        size, name = store_in_shared_memory(output_bytes)
                        self.send_request("send_forward", (module.host, size, name, key))

                        # Clean memory efficiently
                        if self.device.type != "cpu":
                            torch.cuda.empty_cache()

                        # self.store_snapshot(module_id, inp, out, key[0], key[1])

                        if module.training:
                            module.n_batch += 1

    def send_request(self, request_type, args, timeout=None):
        request = {"type": request_type, "args": args}
        try:
            self.mpc_lock.acquire(timeout=timeout)
            self.node_requests.put(request)
            response = self.node_responses.get(timeout=timeout)  # Blocking call, waits for response

        except TimeoutError as e:
            self.terminate = True

        except Exception as e:
            # print(f"Error sending request: {e}")
            response = {"return": str(e)}

        finally:
            self.mpc_lock.release()

        if response:
            return response["return"]

    def store_snapshot(self, module_id, _input, _output, epoch, micro):
        # Ensure the snapshots directory exists
        os.makedirs("tmp/snapshots", exist_ok=True)

        # Get parameters (state_dict) and convert tensors to a serializable format
        params = {k: v.cpu().numpy().tolist() for k, v in self.modules[module_id].state_dict().items()}

        # Prepare snapshot data
        snapshot = {
            "id": module_id,
            "params": params,
            "input": _input.cpu().numpy().tolist(),  # Assuming _input is a tensor
            "output": _output.cpu().numpy().tolist(),  # Assuming _output is a tensor
            "epoch": epoch,
            "micro": micro
        }

        # Define the filename
        file_path = os.path.join("tmp", "snapshots", f"{module_id}_{epoch}_{micro}.json")

        # Write the snapshot to a JSON file
        try:
            with open(file_path, "w") as f:
                json.dump(snapshot, f)
            print(f"Snapshot saved successfully: {file_path}")
        except IOError as e:
            print(f"Error saving snapshot: {e}")

    def load_module(self, file_name, module_id, node_id, module_name, optimizer_name):

        # Load the module in a separate thread
        if len(module_name) > 0:
            api = HfApi()
            try:
                api.model_info(repo_id=module_name)
                state_dict = torch.load(file_name, weights_only=True)
                config = state_dict.pop("module_config")
                module_class = state_dict.pop("module_class")

                architectures = config.get("architectures", [])

                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path=module_name,
                    **config
                )

                if not architectures:
                    raise ValueError("No architecture information found in saved config")

                # Find the appropriate model class
                model_class_name = architectures[0]
                model_class = None
                for task_type, auto_class in MODEL_TYPE_MAPPING.items():
                    if task_type in model_class_name:
                        model_class = auto_class
                        break

                # Default to base model if no specific task type is found
                if model_class is None:
                    model_class = AutoModel

                try:
                    if module_class != model_class_name:
                        model_class = getattr(__import__("transformers"), module_class)
                finally:
                    if hasattr(model_class, "from_config"):
                        module = model_class.from_config(config)
                    else:
                        module = model_class(config)

                try:
                    module.load_state_dict(state_dict)
                except Exception as e:
                    raise e

            except Exception as e:
                # TODO route error to validator for reporting
                return

        else:
            module = torch.jit.load(file_name)

        os.remove(file_name)

        # Initialize queues and states
        module.forward_queue = queue.Queue()
        module.backward_queue = queue.Queue()
        module.intermediates = {}
        module.host = node_id
        module.n_batch = 0

        self.modules[module_id] = module
        self.optimizers[module_id] = get_optimizer_from_name(optimizer_name)
        self.send_request("module_loaded", module_id)

    def check_node(self):
        update_check_interval = 50
        counter = 0

        while not self.terminate:
            if counter % update_check_interval == 0:
                args = self.send_request("check_module", None)

                if isinstance(args, tuple):
                    file_name, module_id, node_id, module_name, optimizer_name = args
                    self.load_module(file_name, module_id, node_id, module_name, optimizer_name)

                # Check for job completion/deletion requests
                elif isinstance(args, str):
                    if args in self.modules:
                        del self.modules[args]
                        del self.optimizers[args]
                        self.send_request("debug_print", (f"Module {args} removed.",))

                # Check for node termination requests
                self.check_for_termination()

            # Process training, forward, and backward queues
            if self.modules:
                for module_id in self.modules.keys():
                    module = self.modules[module_id]

                    # Check if module is in training mode
                    is_training = self.send_request("check_train", module_id)
                    if isinstance(is_training, bool):
                        module.training = is_training

                    # Check for parameters requests
                    params_req = self.send_request("check_parameters_request", module_id)
                    if params_req:
                        self.send_request("debug_print", ("DistributedWorker -> Sending parameters.",))
                        with open(f"parameters_{module_id}", "wb") as file:
                            torch.save(module.state_dict(), file)

                        self.send_request("send_parameters", (module.host, module_id))

                    # Handle forward queue
                    forward_task = self.send_request("check_forward", module_id)
                    if forward_task:
                        module.forward_queue.put(forward_task)

                    # Handle backward queue
                    backward_task = self.send_request("check_backward", module_id)
                    if backward_task:
                        module.backward_queue.put(backward_task)

                    state_update = self.send_request("check_state_update", module_id)
                    if state_update:
                        with self.lock:
                            if state_update[0] == "init":
                                optimizer_kwargs = state_update[1]
                                self.optimizers[module_id] = self.optimizers[module_id](module.parameters(), **optimizer_kwargs)
                                self.send_request("debug_print",
                                                  ("DistributedWorker -> Initialized optimizer.", "bright_blue",
                                                   logging.INFO))
                                self.send_request("optimizer_response", (module_id, "loaded"))

                            elif state_update[0] == "step":
                                closure = state_update[1]
                                self.optimizers[module_id].step(closure)
                                self.send_request("debug_print",
                                                  ("DistributedWorker -> Optimizer stepped.", "bright_blue",
                                                   logging.INFO))
                                self.send_request("optimizer_response", (module_id, "stepped"))

                            elif state_update[0] == "zero_grad":
                                self.optimizers[module_id].zero_grad()
                                self.send_request("debug_print",
                                                  ("DistributedWorker -> Optimizer zeroed.", "bright_blue",
                                                   logging.INFO))
                                self.send_request("optimizer_response", (module_id, "zeroed"))

            counter += 1
            time.sleep(0.2)

    def check_for_termination(self):
        # Send a request to check if the node is shutting down
        shutdown_signal = self.send_request("check_shutdown", None)
        if shutdown_signal:  # Assuming shutdown_signal is True or some indication of shutdown
            self.send_request("debug_print", "Termination signal received. Shutting down DistributedWorker process...")
            self.terminate = True

    def run(self):
        node_check_thread = threading.Thread(target=self.check_node)
        node_check_thread.start()

        self.train_loop()

        node_check_thread.join()

