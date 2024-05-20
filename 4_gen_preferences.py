import argparse
import tqdm as tqdm
from datasets import Dataset
import numpy as np

from src.data import load_sft_dataset

parser = argparse.ArgumentParser(description="Phase 4: Preference Data Generation.")
parser.add_argument("-d", "--dataset", tpye=str, required=True)
parser.add_argument("--data_percentage", type=int, default=1)
parser.add_argument("--run_path", type=str, default="dev", help="Path to save the dataset inside the 'datasets/' folder")
parser.add_argument("--seed", type=int, default=420)
args = parser.parse_args()

args.output_dir = f"datasets/{args.run_name}"
os.makedirs(args.output_dir, exist_ok=True)

dataset = load_sft_dataset(args, percentage=args.data_percentage)

new_dataset = {"prompt": [], "chosen": [], "rejected": [], "score_chosen": [], "score_rejected": []}

for sample in tqdm(dataset.iter(batch_size=4), total=len(dataset)):
    prompt = sample["prompt"][0]
    responses = sample["_response"]
    scores = sample["score"]

    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)

    new_dataset["prompt"].append(prompt)
    new_dataset["chosen"].append(responses[best_idx])
    new_dataset["rejected"].append(responses[worst_idx])
    new_dataset["score_chosen"].append(scores[best_idx])
    new_dataset["score_rejected"].append(scores[worst_idx])

new_dataset = Dataset.from_dict(new_dataset)
new_dataset.to_parquet(f"{args.output_dir}/generated_preferences.parquet")

