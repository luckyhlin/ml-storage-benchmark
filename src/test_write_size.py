from transformers import GPT2Tokenizer, GPT2Model, Trainer, TrainingArguments

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Set training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints_test_write_size",        # Directory where checkpoints will be saved
    save_strategy="epoch",            # Save checkpoint after every epoch
    per_device_train_batch_size=8,    # Batch size
    num_train_epochs=1,               # Number of epochs
    logging_dir="./logs",             # Directory for logs
)

# Initialize Trainer
trainer = Trainer(
    model=model,                      # The model to train
    args=training_args,               # Training arguments
)

# After training finishes, save the final model checkpoint
trainer.save_model("./final_model_checkpoint_test_write_size")
