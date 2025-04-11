import time
import torch
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorlink import DistributedModel


model_name = "microsoft/Phi-4-mini-instruct"

# Parameters for distributed model request.
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1

# Chatbot parameters
MAX_HISTORY_TURNS = 6
MAX_TOKENS = 2048
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

chat_history = deque(maxlen=MAX_HISTORY_TURNS)


if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get distributed model with HF model name
    distributed_model = DistributedModel(
        model_name, training=False, n_pipelines=PIPELINES
    )

    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Append user input to history
        chat_history.append(f"User: {user_input}")

        # Tokenize input
        full_prompt = "\n".join(chat_history) + "\nBot:"
        inputs = tokenizer.encode(
            full_prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
        )

        # Generate response
        with torch.no_grad():
            outputs = distributed_model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,  # still valid, optional if eos_token_id is used
                eos_token_id=tokenizer.eos_token_id,  # explicitly recommended
                do_sample=True,  # temperature only has effect if sampling is on
            )

        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new bot response (naive version — refine if needed)
        bot_reply = response.split("Bot:")[-1].strip()
        print(f"Bot: {bot_reply}\n")
        chat_history.append(f"Bot: {bot_reply}")
