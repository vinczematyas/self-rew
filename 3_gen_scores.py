import argparse
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import os
import re

from src.model import load_model
from src.data import load_sft_dataset

parser = argparse.ArgumentParser(description='Phase 3: Reward Generation')
parser.add_argument("-m", "--model", type=str, default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
parser.add_argument("-d", "--dataset", type=str, default="HuggingFaceH4/deita-10k-v0-sft")
parser.add_argument("--data_percentage", type=int, default=1)
parser.add_argument("--run_path", type=str, default="dev", help="Path to save the dataset inside the 'datasets/' folder")
parser.add_argument("--seed", type=int, default=420)
parser.add_argument("--max_seq_length", type=int, default=4096)
args = parser.parse_args()

args.output_dir = f"datasets/{args.run_name}"
os.makedirs(args.output_dir, exist_ok=True)

# Load model and tokenizer for generation
model, tokenizer = load_model(args)
FastLanguageModel.for_inference(model)

# Load dataset
dataset = load_sft_dataset(args, percentage=args.data_percentage)
if type(dataset) == DatasetDict:
    dataset = dataset["train"]

# load LLM-as-a-Judge prompt
with open('llm_as_a_judge_prompt.txt', 'r') as f:
    DEFAULT_LLM_AS_JUDGE_PROMPT = f.read()
    f.close()

pattern = r"[Ss]core: ([0-5])"
new_dataset = {"prompt": [], "response": [], "score": []}

for sample in tqdm(dataset.iter(batch_size=1)):
    prompt = sample["prompt"]
    response = sample["response"]

    # create formatted prompt for LLM-as-a-Judge
    judge_formatted_prompt = DEFAULT_LLM_AS_JUDGE_PROMPT.format(prompt=prompt, response=response)

    tokenized_judge_formatted_prompt = tokenizer.tokenize(judge_formatted_prompt, return_tensors="pt").to("cuda")

    generated_response = tokenizer.batch_decode(
        model.generate(
            **judge_formatted_prompt,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,  # TODO: do we need this
        )
    )[0]

    # extract score from generated answer
    score = re.findall(pattern, generated_response)

    if score:
        new_dataset["prompt"].append(prompt)
        new_dataset["response"].append(response)
        new_dataset["score"].append(int(score[0]))

# save new dataset
new_dataset = Dataset.from_dict(new_dataset)
new_dataset.to_parquet(f"{args.output_dir}/generated_scores.parquet")

