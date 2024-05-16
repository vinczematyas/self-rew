import argparse
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import TextStreamer
from datasets import Dataset
import re

from src.model import load_model
from src.utils import model_dict, dataset_dict
from src.data import load_sft_dataset

parser = argparse.ArgumentParser(description='Phase 5: Direct Preference Optimization')
parser.add_argument("-m", "--model_name", type=str, default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
parser.add_argument("-d", "--dataset_name", type=str, default="habanoz/lima-chat-format")
parser.add_argument("--data_percentage", type=int, default=1)
parser.add_argument("--run_name", type=str, default="dev")
parser.add_argument("--seed", type=int, default=420)
parser.add_argument("--max_seq_length", type=int, default=2048)
args = parser.parse_args()

assert args.model_name in model_dict and args.dataset_name in dataset_dict

    # Create output directory
args.output_dir = f"models/{model_dict[args.model_name]}/generation/{args.run_name}"

# Load model and tokenizer for generation
model, tokenizer = load_model(args)
FastLanguageModel.for_inference(model)
text_streamer = TextStreamer(tokenizer)

# Load dataset (only prompt-response pairs)
dataset = load_sft_dataset(args, dataset_dict, percentage=args.data_percentage)

# load LLM-as-a-Judge prompt
with open('llm_as_a_judge_prompt.txt', 'r') as f:
    DEFAULT_LLM_AS_JUDGE_PROMPT = f.read()
    f.close()

pattern = r"[Ss]core: ([0-5])"
scores = []
new_dataset = {"prompt": [], "response": [], "score": []}

for sample in tqdm(dataset.iter(batch_size=1)):
    prompt = sample["prompt"]
    response = sample["response"]

    # create formatted prompt for LLM-as-a-Judge
    judge_formatted_prompt = tokenizer(
        DEFAULT_LLM_AS_JUDGE_PROMPT.format(prompt=prompt, response=response),
        return_tensors="pt"
    ).to("cuda")

    generated_answer = tokenizer.batch_decode(
        model.generate(
            **judge_formatted_prompt,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            max_new_tokens=100  # score is at the beginning of the answer
        )
    )[0]

    # extract score from generated answer
    score = re.findall(pattern, generated_answer)

    if score:
        new_dataset["prompt"].append(prompt)
        new_dataset["response"].append(response)
        new_dataset["score"].append(int(score[0]))

# save new dataset
new_dataset = Dataset.from_dict(new_dataset)
new_dataset.save_to_disk(args.output_dir)
new_dataset.push_to_hub("vinczematyas/llm-as-a-judge-scores", private=True)  # ISSUE: with ssh the key does not seem to work

