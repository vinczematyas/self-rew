import torch
import argparse
from datasets import load_dataset, DatasetDict
from trl import DPOTrainer
from peft import PeftModel
from transformers import TrainingArguments, AutoTokenizer
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()

parser = argparse.ArgumentParser(description='Phase 5: Direct Preference Optimization')
parser.add_argument("-m", "--model_name", type=str, default="unsloth/tinyllama-bnb-4bit")
parser.add_argument("-d", "--dataset_name", type=str, default="argilla/dpo-mix-7k")
parser.add_argument("--data_percentage", type=int, default=1)
parser.add_argument("--run_name", type=str, default="dev")
parser.add_argument("--seed", type=int, default=420)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=32)args = parser.parse_args()
args = parser.parse_args()

model_dict = {
    "unsloth/tinyllama-bnb-4bit": "tinyllama"
    "unsloth/llama-3-8b-4bit": "llama-8b"
}
dataset_dict = {
    "argilla/dpo-mix-7k": "dpo-mix-7k"
}

# Create output directory
args.output_dir = f"models/{model_dict[args.model_name]}/{args.run_name}/dpo"

# Load model and tokenizer + LORA
model, tokenizer = load_model(args)
peft_model = get_peft_model(args, model)

# Lode dataset
dataset = load_dpo_dataset(args, dataset_dict, percentage=args.data_percentage)

# Train model
trainer = get_dpo_trainer(args, peft_model, tokenizer, dataset)
trainer.train()
trainer.model.save_pretrained(f"{args.output_dir}/checkpoint_final")
