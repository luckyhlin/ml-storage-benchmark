import time
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from datasets import load_dataset
import torch
import sys

# # Step 1: Define the Quantization Configuration
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
# )

# Step 2: Load the Quantized GPT-2 Model
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=quantization_config,
    device_map="auto",
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fix Padding Issues
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Step 3: Load and Preprocess the Dataset
def profile_loading(examples):
    text = " ".join(examples["tokens"])
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    tokenized_dict = {
        "input_ids": tokenized["input_ids"].squeeze(0).tolist(),
        "attention_mask": tokenized["attention_mask"].squeeze(0).tolist(),
    }
    return tokenized_dict

dataset = load_dataset("wikiann", "en", split="train")

# Calculate Dataset Size
def get_dataset_size(dataset):
    total_bytes = 0
    for example in dataset:
        total_bytes += sys.getsizeof(example)
    return total_bytes

num_samples = len(dataset)
dataset_size_bytes = get_dataset_size(dataset)
print(f"Dataset contains {num_samples} examples.")
print(f"Estimated dataset size: {dataset_size_bytes / (1024 ** 2):.2f} MB")

dataset = dataset.map(profile_loading, remove_columns=["tokens"])

# For causal language modeling, the labels are the same as input_ids
def add_labels(example):
    example["labels"] = example["input_ids"]
    return example

dataset = dataset.map(add_labels, batched=False)

# Use a small subset for demonstration if desired (comment out to use full)
# dataset = dataset.select(range(1000))

# Split dataset for training and evaluation
split = dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

# Step 4: Create a Data Collator
# For causal LM, we can use the DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=1,           # Increase as needed
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,              # Adjust as needed
    logging_steps=100,
    eval_steps=500,              # Evaluate every 500 steps
    learning_rate=1e-4,
    fp16=True,
    report_to="none",            # Change this to "wandb" or "tensorboard" if needed
    push_to_hub=False,
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Step 7: Train and Save Checkpoints
trainer.train()

# After training finishes, save the final model checkpoint
trainer.save_model("./final_model_checkpoint")

print("Training completed and model checkpoint saved.")
