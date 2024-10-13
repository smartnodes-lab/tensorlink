from tensorlink.ml.distributed import DistributedModel

import time
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from datasets import load_dataset
from tqdm import tqdm
from transformers import set_seed, get_linear_schedule_with_warmup
import logging


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def preprocess_data(dataset, tokenizer, split, logger):
    input_ids = []
    attention_masks = []

    logger.info("Preprocessing data")
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


def train(model, tokenizer, device, logger, batch_size):
    epochs = 1
    set_seed(42)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(512))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(128))
    train_dataset = preprocess_data(dataset, tokenizer, "train", logger)
    val_dataset = preprocess_data(dataset, tokenizer, "validation", logger)

    logger.info("Train dataset size: %d", len(train_dataset))
    logger.info("Validation dataset size: %d", len(val_dataset))

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
    total_start_time = time.time()

    for epoch_i in range(epochs):
        start_time = time.time()
        logger.info("======== Epoch %d / %d ========", epoch_i + 1, epochs)
        logger.info("Training...")

        total_train_loss = 0
        losses = []
        model.train()

        for step, batch in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            optimizer.zero_grad()

            output = model(
                b_input_ids,
                # token_type_ids=None,
                # attention_mask=b_input_mask,
                # labels=b_labels,
            )

            loss = output.loss
            total_train_loss += loss.item()

            losses.append(loss.item())

            if step % 50 == 0 and step != 0:
                logger.info("Step %d average loss: %f", step, np.mean(losses))

            if isinstance(model, DistributedModel):
                model.backward(loss)
            else:
                loss.backward()

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_time = time.time() - start_time
        logger.info("Average training loss: %f", avg_train_loss)
        logger.info("Epoch training time: %.2f seconds", epoch_time)
        logger.info("End of Epoch %d", epoch_i + 1)

        total_eval_accuracy = 0
        total_eval_loss = 0
        logger.info("Running Evaluation")
        model.eval()

        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
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

        logger.info("Validation Loss: %f", avg_val_loss)
        logger.info("Validation Accuracy: %f", avg_val_accuracy)

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
            }
        )

    total_training_time = time.time() - total_start_time
    logger.info("Training complete!")
    logger.info("Total training time: %.2f seconds", total_training_time)

    # Total summary
    avg_train_loss_all_epochs = np.mean([stat["Training Loss"] for stat in training_stats])
    avg_val_loss_all_epochs = np.mean([stat["Valid. Loss"] for stat in training_stats])
    avg_val_accuracy_all_epochs = np.mean([stat["Valid. Accur."] for stat in training_stats])

    logger.info("==== Train Summary ====")
    logger.info("Average Training Loss: %f", avg_train_loss_all_epochs)
    logger.info("Average Validation Loss: %f", avg_val_loss_all_epochs)
    logger.info("Average Validation Accuracy: %f", avg_val_accuracy_all_epochs)
    logger.info("Training stats: %s", training_stats)

    return training_stats