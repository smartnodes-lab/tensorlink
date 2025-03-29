from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.utils import estimate_hf_model_memory, get_hf_model

import threading
import json
import time


MODELS_PATH = "logs/models.json"


def load_models():
    try:
        with open(MODELS_PATH, "r") as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_models(models):
    with open(MODELS_PATH, "w") as f:
        json.dump(models, f, indent=4)


class DistributedValidator:
    def __init__(self, node_requests, node_responses, mpc_lock, trusted=False):
        self.node_requests = node_requests
        self.node_responses = node_responses
        self.mpc_lock = mpc_lock
        self.terminate = False
        self.models = load_models()

    def send_request(self, request_type, args, timeout=None):
        request = {"type": request_type, "args": args}
        response = None

        try:
            self.mpc_lock.acquire(timeout=timeout)
            self.node_requests.put(request)
            response = self.node_responses.get(
                timeout=timeout
            )  # Blocking call, waits for response

        except TimeoutError as e:
            self.terminate = True

        except Exception as e:
            response = {"return": str(e)}

        finally:
            self.mpc_lock.release()

        if response:
            return response["return"]

    def check_node(self):
        counter = 0

        while not self.terminate:
            # if counter % update_check_interval == 0:
            # args = self.send_request("check_pol", None)
            #
            # # If we have any PoL checks to attend to...
            # if args:
            #     pass

            # If we have any directly-hosted/api jobs to attend to
            job_data = self.send_request("get_jobs", None)

            if isinstance(job_data, dict):
                """
                job_data = {
                    "author": requesters_ip,
                    "active": False,
                    "hosted": True,
                    "ram": ram,
                    "vram": vram,
                    "time": _time,
                    "payment": job_info.get("payment", 0),
                    "n_pipelines": 1,
                    "dp_factor": 1,
                    "distribution": {},
                    "n_workers": 0,
                    "seed_validators": [self.rsa_key_hash]
                }
                """
                if job_data.get("vram", 0) < 8e9:
                    model_name = job_data.get("model_name")

                    if model_name in self.models:
                        distribution = job_data["distribution"] = self.models[
                            model_name
                        ].get("distribution", {})

                    else:
                        # Load HF model, create and save distribution.
                        model, tokenizer = get_hf_model(model_name, tokenizer=True)
                        parser = ModelParser()
                        distribution = parser.create_distributed_config(
                            model, training=False, trusted=False
                        )
                        self.models[model_name] = {"distribution": distribution}
                        save_models(self.models)

                    job_data["distribution"] = distribution
                    self.send_request("send_hosted_job_request", job_data)

            time.sleep(3)

    def run(self):
        node_check_thread = threading.Thread(target=self.check_node)
        node_check_thread.start()
        node_check_thread.join()
