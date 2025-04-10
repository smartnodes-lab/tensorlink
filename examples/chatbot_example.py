import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorlink import DistributedModel

# Parameters for distributed model request.
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1

if __name__ == "__main__":
    # Load tokenizer
    model_name = "microsoft/Phi-4-mini-instruct"  # Change as needed
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
        input_text = user_input
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        # Concatenate chat history here...

        # Generate response
        with torch.no_grad():
            outputs = distributed_model.generate(
                inputs,
                max_length=256,
                max_new_tokens=256,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Bot: {response}\n")
