import requests


https_serv = (
    "https://smartnodes-lab.ddns.net/tensorlink-api"  # May not work with all clients
)
http_serv = "http://smartnodes-lab.ddns.net/tensorlink-api"  # Use this if HTTPS fails


payload = {
    "hf_name": "Qwen/Qwen2.5-7B-Instruct",
    "message": "Describe the role of AI in medicine.",
    "max_length": 1024,
    "max_new_tokens": 256,
    "temperature": 0.7,
    "do_sample": True,
    "num_beams": 4,
    "history": [
        {"role": "user", "content": "What is artificial intelligence?"},
        {"role": "assistant", "content": "Artificial intelligence refers to..."},
    ],
}


response = requests.post(f"{https_serv}/generate", json=payload)
print(response.json())
