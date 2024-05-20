import argparse
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from unsloth import FastLanguageModel
import os

from src.model import load_model
from src.data import load_sft_dataset

parser = argparse.ArgumentParser(description="Phase 2: Response Generation")
parser.add_argument("-m", "--model", type=str, default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
parser.add_argument("-d", "--dataset", type=str, default="HuggingFaceH4/deita-10k-v0-sft")
parser.add_argument("--data_percentage", type=int, default=1)
parser.add_argument("--run_path", type=str, default="dev", help="Path to save the dataset inside the 'datasets/' folder")
parser.add_argument("--seed", type=int, default=420)
parser.add_argument("--max_seq_length", type=int, default=4096)
args = parser.parse_args()

args.output_dir = f"datasets/{args.run_name}"
os.makedirs(args.output_dir, exist_ok=True)

# Load model and tokenizer for inference
model, tokenizer = load_model(args)
FastLanguageModel.for_inference(model)

# Load dataset
dataset = load_sft_dataset(args, percentage=args.data_percentage).shuffle(seed=args.seed)
if type(dataset) == DatasetDict:
    dataset = dataset["train"]

new_dataset = {"prompt": [], "response": []}
for sample in tqdm(dataset.iter(batch_size=1), total=len(dataset)):
    prompt = sample["prompt"]

    tokenized_prompt = tokenizer.tokenize(prompt, return_tensors="pt").to("cuda")

    for _ in range(4):
        generated_response = tokenizer.batch_decode(
            model.generate(
                **tokenized_prompt,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,  # softmax temperature
                top_p = 0.9,  # use only top logits (top probability)
            )
        )[0]

        print(generated_response)
        assert False, "check if trimming is needed and how"

        # FIX:
        # TODO: trim according to https://github.com/Oxen-AI/Self-Rewarding-Language-Models/blob/main/scripts/02_gen_responses.py
        # FIX:

        new_dataset["prompt"].append(prompt)
        new_dataset["response"].append(generated_response)

new_dataset = Dataset.from_dict(new_dataset)
new_dataset.to_parquet(f"{args.output_dir}/generated_responses.parquet")

