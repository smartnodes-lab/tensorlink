from src.coordinator import DistributedCoordinator, WorkerCoordinator, ValidatorCoordinator
import torch
import json
import copy
import time
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm
from multiprocessing import shared_memory


# https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
# https://mccormickml.com/2019/07/22/BERT-fine-tuning/


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def preprocess_data(dataset, tokenizer, split):
    input_ids = []
    attention_masks = []

    print("Preprocessing data")
    for sent in tqdm(dataset[split]):
        encoded_dict = tokenizer.encode_plus(
            sent["text"],
            add_special_tokens=True,
            max_length=100,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(dataset[split]["label"])

    return TensorDataset(input_ids, attention_masks, labels)


def train(model, tokenizer, device):
    batch_size = 32
    epochs = 2
    set_seed(42)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

    # temporarily truncating for testing purposes
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(256))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(64))
    train_dataset = preprocess_data(dataset, tokenizer, "train")
    val_dataset = preprocess_data(dataset, tokenizer, "validation")

    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(val_dataset))

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )

    val_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size
    )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value in run_glue.py
        num_training_steps=total_steps,
    )

    training_stats = []

    for epoch_i in tqdm(range(epochs)):

        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")

        total_train_loss = 0

        model.train()

        losses = []
        print(f"Training Epoch {epoch_i + 1}")
        for step, batch in enumerate(tqdm(train_dataloader, position=0, leave=True)):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            optimizer.zero_grad()

            output = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            loss = output.loss
            total_train_loss += loss.item()
            losses.append(loss.item())
            if step % 50 == 0 and step != 0:
                print(f"Step {step} average loss:", np.mean(losses))

            # if distributed:
            model.backward(loss)
            # else:
            # loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # not sure if this would work with distributed

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print("Average training loss: ", avg_train_loss)
        total_eval_accuracy = 0
        total_eval_loss = 0

        print("Running Evaluation")
        model.eval()

        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # with torch.no_grad():  # idk if this would work with distributed
            output = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            total_eval_loss += output.loss.item()
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)
            avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
            avg_val_loss = total_eval_loss / len(val_dataloader)

        print("Validation Loss: ", avg_val_loss)
        print("Validation Accuracy: ", avg_val_accuracy)

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
            }
        )
        print(training_stats)

    return training_stats


if __name__ == "__main__":
    # Launch Nodes
    user = DistributedCoordinator(debug=True)
    time.sleep(0.2)
    worker = WorkerCoordinator(debug=True)
    time.sleep(0.2)
    validator = ValidatorCoordinator(debug=True)

    # Bootstrap nodes
    val_key, val_host, val_port = validator.send_request("info", None)
    worker.send_request("connect_node", (val_key, val_host, val_port))
    user.send_request("connect_node", (val_key, val_host, val_port))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    ).to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    distributed_model = user.create_distributed_model(model, 1, 1.4e9)
    train(distributed_model, tokenizer, device)
