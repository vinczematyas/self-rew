import os
import torch
import argparse
import pandas as pd
from termcolor import colored
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer

from src.model import load_model, get_peft_model
from src.trainer import get_sft_trainer
from src.data import load_sft_dataset

parser = argparse.ArgumentParser(description="Phase 0: Supervised Fine-Tuning.")
parser.add_argument("-m", "--model_name", type=str, default="unsloth/tinyllama-bnb-4bit")
parser.add_argument("-d", "--dataset_name", type=str, default="HuggingFaceH4/deita-10k-v0-sft")
parser.add_argument("--data_percentage", type=int, default=1)
parser.add_argument("--run_name", type=str, default="dev")
parser.add_argument("--seed", type=int, default=420)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=32)

args = parser.parse_args()

# Model and dataset dictionaries for easy access
model_dict = {
    "unsloth/tinyllama-bnb-4bit": "tinyllama",
    "unsloth/llama-3-8b-bnb-4bit": "llama-8b",
}
dataset_dict = {
    "HuggingFaceH4/deita-10k-v0-sft": "deita-10k-v0-sft",
}

# Create output directory
args.output_dir = f"models/{model_dict[args.model_name]}/{args.run_name}/sft"

# Load model and tokenizer + LORA
model, tokenizer = load_model(args)
peft_model = get_peft_model(args, model)

# Load dataset
dataset = load_sft_dataset(args, dataset_dict, percentage=1)

# Train model
trainer = get_sft_trainer(args, model, tokenizer, dataset)
trainer.train()
trainer.model.save_pretrained(args.output_dir)

