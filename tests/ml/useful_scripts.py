import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, set_seed
from datasets import load_dataset
from tqdm import tqdm
import time
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def preprocess_data(dataset, tokenizer, max_length=128):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    # Directly apply the tokenization function using a smaller dataset
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset


# Function to load and preprocess a subset
def load_and_preprocess_subset(dataset, tokenizer, split, subset_size, max_length=128):
    subset = dataset[split].shuffle(seed=42).select(range(subset_size))
    return preprocess_data(subset, tokenizer, max_length=max_length)


def train(model, optimizer, tokenizer, device, batch_size=8, epochs=1):
    set_seed(42)
    model.to(device)

    # Load a dummy dataset
    dataset = load_dataset("imdb")

    train_subset_size = 500
    test_subset_size = 100

    # Preprocess only the smaller subsets
    train_dataset = load_and_preprocess_subset(dataset, tokenizer, "train", train_subset_size)
    val_dataset = load_and_preprocess_subset(dataset, tokenizer, "test", test_subset_size)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Rest of the training loop
    for epoch_i in range(epochs):
        logger.info(f"Epoch {epoch_i + 1}/{epochs}")
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info(f"Average Training Loss: {avg_train_loss}")

        # Validation
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in tqdm(val_dataloader, desc="Evaluating"):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_eval_loss += loss.item()

                logits = outputs.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_val_loss = total_eval_loss / len(val_dataloader)

        logger.info(f"Validation Loss: {avg_val_loss}")
        logger.info(f"Validation Accuracy: {avg_val_accuracy}")

    logger.info("Training complete!")


# def prepare_dummy_data(choice):
#     texts = ["This is a positive sentence.", "This is a negative sentence."]
#     labels = [1, 0]
#
#     # Tokenization
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
#
#     # dataset
#     if choice == 1:
#         dataset = TensorDataset(tokenized_texts['input_ids'], tokenized_texts['attention_mask'], torch.tensor(labels))
#         dataloader = DataLoader(dataset)
#         return dataloader
#     elif choice == 2:
#         dataset = Dataset.from_dict({
#             'input_ids': tokenized_texts['input_ids'].tolist(),
#             'attention_mask': tokenized_texts['attention_mask'].tolist(),
#             'labels': labels,
#         })
#         return dataset
#     else:
#         return tokenized_texts


# def simple_train(model, optimizer):
#     loss_fn = torch.nn.CrossEntropyLoss()
#
#     dataloader = prepare_dummy_data(1)
#
#     # simple training loop
#     for batch in dataloader:
#         optimizer.zero_grad()
#         input_ids, attention_mask, label = batch
#         outputs = model(input_ids, attention_mask=attention_mask, labels=label)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         print(loss.item())


# def d_simple_train(model, optimizer):
#     loss_fn = torch.nn.CrossEntropyLoss()
#
#     dataloader = prepare_dummy_data(1)
#
#     # simple training loop
#     for batch in dataloader:
#         optimizer.zero_grad()
#         input_ids, attention_mask, label = batch
#         outputs = model(input_ids, attention_mask=attention_mask, labels=label)
#         loss = outputs.loss
#         model.backward(loss)
#         optimizer.step()
#         print(loss.item())


# def hf_train(model, optimizer):
#     optimizer.zero_grad()
#     loss_fn = torch.nn.CrossEntropyLoss()
#
#     training_args = TrainingArguments(
#         output_dir='./results',
#         num_train_epochs=1,
#         per_device_train_batch_size=1,
#         logging_steps=1,
#         overwrite_output_dir=True,
#     )
#
#     dataset = prepare_dummy_data(2)
#
#     def compute_metrics(pred):
#         labels = pred.label_ids
#         preds = pred.predictions.argmax(-1)
#         accuracy = (preds == labels).mean()
#         loss = pred.loss
#         return {"accuracy": accuracy}
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#         compute_metrics=compute_metrics
#     )
#
#     trainer.train()


# def fp_check(model):
#     tokenized_texts = prepare_dummy_data(3)
#     input_ids = tokenized_texts['input_ids'][0].unsqueeze(0)  # Add batch dimension
#     attention_mask = tokenized_texts['attention_mask'][0].unsqueeze(0)  # Add batch dimension
#
#     # Perform a forward pass through the model
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)
#
#     print(outputs.logits)
