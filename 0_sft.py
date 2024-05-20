import os
import argparse
from datasets import DatasetDict

from src.model import load_model, get_peft_model
from src.trainer import get_sft_trainer
from src.data import load_sft_dataset

parser = argparse.ArgumentParser(description="Phase 0: Supervised Fine-Tuning.")
parser.add_argument("-m", "--model", type=str, default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
parser.add_argument("-d", "--dataset", type=str, default="HuggingFaceH4/deita-10k-v0-sft")
parser.add_argument("--data_percentage", type=int, default=1)
parser.add_argument("--run_path", type=str, default="dev", help="Path to save the model inside the 'models/' folder")
parser.add_argument("--seed", type=int, default=420)
parser.add_argument("--max_seq_length", type=int, default=4098)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=32)
args = parser.parse_args()

args.output_dir = f"models/{args.run_path}"
os.makedirs(args.output_dir, exist_ok=True)

# Load model and tokenizer + LORA
model, tokenizer = load_model(args)
peft_model = get_peft_model(args, model)

# Load dataset
dataset = load_sft_dataset(args, percentage=args.data_percentage).shuffle(seed=args.seed)
assert type(dataset) == DatasetDict, "Dataset must be a DatasetDict with 'train' and 'test' keys."

# Train model
trainer = get_sft_trainer(args, peft_model, tokenizer, dataset)
trainer.train()

# Save model
model.save_pretrained(f"{args.output_dir}/checkpoint-final")
tokenizer.save_pretrained(f"{args.output_dir}/checkpoint-final")

