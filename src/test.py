import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import sys

# Step 1: Define the Quantization Configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Step 2: Load the Quantized GPT-2 Model
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
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

# Step 4: Create a DataLoader with simplified collate_fn
def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}

data_loader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collate_fn,
)

# Step 5: Perform Inference
device = next(model.parameters()).device

for batch in data_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,
        top_p=0.9,
    )

    for output in outputs:
        print("Generated Text:", tokenizer.decode(output, skip_special_tokens=True))
    # break  # Remove this break to process the entire dataset
