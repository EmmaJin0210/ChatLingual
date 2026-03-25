import json
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from src.core.models.model_constants import (
    MODEL_ID_BP,
    ROOT_MODELS,
    ROOT_SENTENCES_JPN,
    FILENAME_TRAIN_DATA,
    FILENAME_EVAL_DATA,
)

# Initialize WandB
wandb.init(project="fudge-bp-training", name="modernbert-wjt")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_BP)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID_BP, num_labels=5)

def flatten_label(example):
    label_map = {"n1": 0, "n2": 1, "n3": 2, "n4": 3, "n5": 4}
    if isinstance(example.get("label"), list):
        label_val = example["label"][0]
    else:
        label_val = example["label"]
    if isinstance(label_val, str):
        example["label"] = label_map[label_val]
    return example

def tokenize_function(examples):
    return tokenizer(
        examples["prefix"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

def load_dataset_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    dataset = dataset.map(flatten_label)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


train_dataset = load_dataset_from_json(f"{ROOT_SENTENCES_JPN}{FILENAME_TRAIN_DATA}")
eval_dataset = load_dataset_from_json(f"{ROOT_SENTENCES_JPN}{FILENAME_EVAL_DATA}")


labels = np.array(train_dataset["labels"])
class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels]


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=False
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=sampler,
    collate_fn=data_collator,
    num_workers=4
)

# Training arguments
training_args = TrainingArguments(
    output_dir=f"{ROOT_MODELS}modernbert-predictor",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="/scratch/anon/.logs",
    logging_steps=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="wandb"
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)


def custom_get_train_dataloader(trainer):
    return train_loader

trainer.get_train_dataloader = lambda: custom_get_train_dataloader(trainer)


trainer.train()

wandb.finish()
