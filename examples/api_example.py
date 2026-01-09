import requests


HTTPS_SERV = (
    "https://smartnodes.ddns.net/tensorlink-api"  # May not work with all clients
)
HTTP_SERV = "http://smartnodes.ddns.net/tensorlink-api"  # Use this if HTTPS fails


def generate():
    payload = {
        "hf_name": "Qwen/Qwen3-8B",
        "message": "Describe the role of AI in medicine.",
        "max_length": 4096,
        "max_new_tokens": 4096,
        "temperature": 0.7,
        "do_sample": True,
        "num_beams": 3,
        "history": [
            {"role": "user", "content": "What is artificial intelligence?"},
            {"role": "assistant", "content": "Artificial intelligence refers to..."},
        ],
    }

    response = requests.post(f"{HTTPS_SERV}/generate", json=payload)
    print(response.json())


def request_model():
    payload = {"hf_name": "Qwen/Qwen3-8B"}
    response = requests.post(f"{HTTPS_SERV}/request-job", json=payload)
    response.json()
