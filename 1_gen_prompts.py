import argparse
from tqdm import tqdm
from datasets import Dataset
from unsloth import FastLanguageModel

# from src.model import load_model
from src.data import load_sft_dataset
from src.utils import model_dict, dataset_dict

parser = argparse.ArgumentParser(description='Phase 5: Direct Preference Optimization')
parser.add_argument("-m", "--model_name", type=str, default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
parser.add_argument("-d", "--dataset_name", type=str, default="habanoz/lima-chat-format")
parser.add_argument("--data_percentage", type=int, default=1)
parser.add_argument("--run_name", type=str, default="dev")
parser.add_argument("--seed", type=int, default=420)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--n_shot", type=int, default=8)
args = parser.parse_args()

assert args.model_name in model_dict and args.dataset_name in dataset_dict

# Create output directory
args.output_dir = f"models/{model_dict[args.model_name]}/generation/{args.run_name}"

# Load model and tokenizer for generation
model, tokenizer = load_model(args)
FastLanguageModel.for_inference(model)

# Load dataset (only prompt-response pairs)
dataset = load_sft_dataset(args, dataset_dict, percentage=args.data_percentage).shuffle(seed=args.seed)

new_dataset = {"generated_prompt": []}

for sample in tqdm(dataset.iter(batch_size=args.n_shot), total=len(dataset)):
    n_shot_prompt = """\
    Come up with a series of tasks and questions. 
    Only the task/question, no further text/explanation, no additional information.
    The task or question should be something a person would ask a chatbot.
    """
    for example_prompt in sample["prompt"]:
        n_shot_prompt += f"\n<task>{example_prompt}</task>"

    print(n_shot_prompt)
    tokenized_n_shot_prompt = tokenizer(n_shot_prompt, return_tensors="pt").to("cuda")

    new_prompt = tokenizer.batch_decode(
        model.generate(
            **tokenized_n_shot_prompt,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            max_new_tokens=100,  # TODO: change this to a reasonable value
        )
    )[0]

    # TODO: any timming needed??

    new_dataset["generated_prompt"].append(new_prompt)

new_dataset = Dataset.from_dict(new_dataset)
new_dataset.save_to_disk(args.output_dir)
new_dataset.push_to_hub("vinczematyas/srlm/generated-prompts")

