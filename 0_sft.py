import os
import argparse
import pandas as pd
from termcolor import colored

from src.model import load_model, get_peft_model
from src.trainer import get_sft_trainer
from src.data import load_alpaca_ds


def run(args):
    # load model and tokenizer using Unsloth
    model, tokenizer = load_model(args.model_name)

    # load and tokenize dataset
    ds = load_alpaca_ds(tokenizer)

    # get LORA and SFTTrainer
    model = get_peft_model(model)
    trainer = get_sft_trainer(model, tokenizer, ds, args.output_dir, args.seed)

    assert False, "waiting for GPU"  # ISSUE:

    trainer.train()
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))
    print(colored("Final SFT checkpoint saved", "cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Supervised Fine-Tuning.")
    parser.add_argument("-m", "--model_name", type=str, default="unsloth/tinyllama-bnb-4bit")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()

    model_dict = {
        "unsloth/tinyllama-bnb-4bit": "tinyllama"
    }

    args.output_dir = f"models/{model_dict[args.model_name]}/{args.run_name}/sft"

    run(args)
