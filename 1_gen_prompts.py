import argparse
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from unsloth import FastLanguageModel
import os

from src.model import load_model
from src.data import load_sft_dataset

parser = argparse.ArgumentParser(description="Phase 1: Prompt Generation")
parser.add_argument("-m", "--model", type=str, default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
parser.add_argument("-d", "--dataset", type=str, default="HuggingFaceH4/deita-10k-v0-sft")
parser.add_argument("--data_percentage", type=int, default=1)
parser.add_argument("--run_path", type=str, default="dev", help="Path to save the dataset inside the 'datasets/' folder")
parser.add_argument("--seed", type=int, default=420)
parser.add_argument("--max_seq_length", type=int, default=4096)
parser.add_argument("--n_shot", type=int, default=2, help="Number of random examples per prompt generation")
args = parser.parse_args()

args.output_dir = f"datasets/{args.run_path}"
os.makedirs(args.output_dir, exist_ok=True)

# Load model and tokenizer for inference
model, tokenizer = load_model(args)
FastLanguageModel.for_inference(model)

# Load dataset
dataset = load_sft_dataset(args, percentage=args.data_percentage).shuffle(seed=args.seed)
if type(dataset) == DatasetDict:
    dataset = dataset["train"]

new_dataset = {"prompt": []}
for sample in tqdm(dataset.iter(batch_size=args.n_shot), total=len(dataset)//args.n_shot):
    n_shot_prompt = """\
    Come up with a series of tasks and questions. 
    Only the task/question, no further text/explanation, no additional information.
    The task or question should be something a person would ask a chatbot.
    """
    for example_prompt in sample["prompt"]:
        n_shot_prompt += f"\n<task>{example_prompt}</task>"

    tokenized_n_shot_prompt = tokenizer(n_shot_prompt, return_tensors="pt").to("cuda")

    new_prompts = tokenizer.batch_decode(
        model.generate(
            **tokenized_n_shot_prompt,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,  # TODO: do we need this
        )
    )[0]

    while True:
        start = new_prompts.find("<task>")
        end = new_prompts.find("</task>")
        if start == -1 or end == -1:
            break
        new_dataset["prompt"].append(new_prompts[start+6:end])
        new_prompts = new_prompts[end+7:]

# Save the generated prompts
new_dataset = Dataset.from_dict(new_dataset)
new_dataset.to_parquet(f"{args.output_dir}/generated_prompts.parquet")

