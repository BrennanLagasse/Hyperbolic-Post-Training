# Currently using vllm env with this

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import wandb

# =========================
# Config
# =========================
MODEL_NAME = "Qwen/Qwen3-1.7B"
DATASET_NAME = "Open-Orca/OpenOrca"
OUTPUT_DIR = "./qwen3-openorca-ft"
MAX_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-5
EPOCHS = 1
SAVE_STEPS = 500
LOGGING_STEPS = 50

# =========================
# Init W&B
# =========================
wandb.init(project="qwen3-openorca-ft")

# =========================
# Load tokenizer & model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# =========================
# Load dataset
# =========================
dataset = load_dataset(DATASET_NAME)

# =========================
# Formatting function
# =========================
def format_example(example):
    prompt = example.get("question", "")
    response = example.get("response", "")

    text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# =========================
# Tokenize dataset
# =========================
tokenized_dataset = dataset["train"].map(
    format_example,
    remove_columns=dataset["train"].column_names,
    num_proc=4
)

# =========================
# Data collator
# =========================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# =========================
# Training args
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    fp16=True,
    report_to="wandb",
    run_name="qwen3-openorca",
    logging_dir=f"{OUTPUT_DIR}/logs",
    dataloader_num_workers=4,
    save_strategy="steps",
    evaluation_strategy="no"
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# =========================
# Resume if checkpoint exists
# =========================
checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-")
    ]
    if len(checkpoints) > 0:
        checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Resuming from checkpoint: {checkpoint}")

# =========================
# Train
# =========================
trainer.train(resume_from_checkpoint=checkpoint)

# =========================
# Save final model
# =========================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

wandb.finish()

print("Training complete.")
