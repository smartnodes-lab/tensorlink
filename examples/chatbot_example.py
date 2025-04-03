import time
import torch
from transformers import AutoTokenizer
from tensorlink import DistributedModel

# Parameters for distributed model request.
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1

if __name__ == "__main__":
    # Load tokenizer
    model_name = (
        'bert-base-uncased'  # TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change as needed
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get distributed model
    distributed_model = DistributedModel(
        model_name, training=False, n_pipelines=PIPELINES
    )

    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt")

        # Generate response
        with torch.no_grad():
            output_tokens = distributed_model.generate(**inputs, max_length=100)

        # Decode and print response
        response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(f"Bot: {response}\n")
