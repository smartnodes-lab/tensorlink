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
            # Other code...

            job_data = self.send_request("get_jobs", None)

            if isinstance(job_data, dict):
                if job_data.get("vram", 0) < 8e9:
                    model_name = job_data.get("model_name")

                    # Check if we already have the distribution saved
                    if (
                        model_name in self.models
                        and "distribution" in self.models[model_name]
                    ):
                        # Use the saved distribution
                        distribution = self.models[model_name]["distribution"]
                        job_data["distribution"] = distribution

                        # Update RAM and VRAM estimates from saved data
                        job_data["vram"] = sum(
                            [v["size"] for v in distribution.values()]
                        )
                        job_data["ram"] = sum(
                            [v["size"] for v in distribution.values()]
                        )
                    else:
                        # Load HF model, create and save distribution
                        model, tokenizer = get_hf_model(model_name, tokenizer=True)
                        parser = ModelParser()
                        distribution = parser.create_distributed_config(
                            model,
                            training=job_data.get("training", False),
                            trusted=False,
                        )

                        # Save the distribution
                        self.models[model_name] = {"distribution": distribution}
                        save_models(self.models)

                        job_data["distribution"] = distribution
                        job_data["vram"] = sum(
                            [v["size"] for v in distribution.values()]
                        )
                        job_data["ram"] = sum(
                            [v["size"] for v in distribution.values()]
                        )

                    # Process the job with the distribution (whether loaded or cached)
                    if job_data["hosted"]:
                        self.send_request("send_hosted_job_request", job_data)
                    else:
                        try:
                            self.send_request("send_hf_job_request", job_data)
                        except Exception as e:
                            print(str(e))

            time.sleep(3)

    def run(self):
        node_check_thread = threading.Thread(target=self.check_node)
        node_check_thread.start()
        node_check_thread.join()
