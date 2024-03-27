from src.cryptography.rsa import get_rsa_pub_key
from src.roles.worker import Worker
from src.ml.distributed import DistributedModel
import torch
import json
import copy
import time
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer, set_seed, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm

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
            sent['text'],
            add_special_tokens = True,
            max_length = 100,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(dataset[split]['label'])

    return TensorDataset(input_ids, attention_masks, labels)

def train(model, tokenizer, device, distributed=False):
    batch_size = 16
    epochs = 2
    set_seed(42)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    # temporarily truncating for testing purposes
    dataset['train'] = dataset['train'].shuffle(seed=42).select(range(100))
    dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(10))
    train_dataset = preprocess_data(dataset, tokenizer, 'train')
    val_dataset = preprocess_data(dataset, tokenizer, 'validation')

    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(val_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    val_dataloader = DataLoader(
        val_dataset,
        sampler = SequentialSampler(val_dataset),
        batch_size = batch_size 
    )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
        num_warmup_steps = 0, # Default value in run_glue.py
        num_training_steps = total_steps)
    
    training_stats = []
    
    for epoch_i in tqdm(range(epochs)):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        total_train_loss = 0

        if distributed:
            pass
        else:
            model.train() # case would be different for distributed model

        losses = []
        print(f"Training Epoch {epoch_i + 1}")
        for step, batch in enumerate(tqdm(train_dataloader)):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            optimizer.zero_grad()

            output = model(b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels)
            
            loss = output.loss
            total_train_loss += loss.item()
            losses.append(loss.item())
            if step % 50 == 0 and step != 0:
                print(f"Step {step} average loss:", np.mean(losses))

            if distributed:
                model.backward(loss)
            else:
                loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # not sure if this would work with distributed    

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print("Average training loss: ", avg_train_loss)
        total_eval_accuracy = 0
        total_eval_loss = 0

        print("Running Evaluation")
        if distributed:
            pass
        else:
            model.eval()

        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad(): # idk if this would work with distributed
                output = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels)
                
            total_eval_loss += output.loss.item()
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)
            avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
            avg_val_loss = total_eval_loss / len(val_dataloader)  

        print("Validation Loss: ", avg_val_loss)
        print("Validation Accuracy: ", avg_val_accuracy)

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
            }
        )
        print(training_stats)
        
    return training_stats  


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    ip = "127.0.0.1"
    port = 5026

    # Spawn 3 workers on their own ports + threads
    worker1 = Worker(host=ip, port=port, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
                     debug=True)
    worker2 = Worker(host=ip, port=port + 1, wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
                     debug=True)

    worker1.master = True  # We must omit this
    worker2.training = True
    worker1.connect_dht_node(ip, port + 1)
    config = {"encoder": worker1.key_hash}


    train(model, tokenizer, device)




