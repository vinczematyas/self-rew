from datasets import load_dataset, DatasetDict


def load_sft_dataset(args, dataset_dict, percentage = 1):
    if dataset_dict[args.dataset_name] == "deita-10k-v0-sft":
        dataset = load_dataset(
            args.dataset_name, 
            split=[f"train_sft[:{percentage}%]", f"test_sft[:{percentage}%]"]
        )
        dataset = DatasetDict({"train": dataset[0], "test": dataset[1]})
    return dataset

def load_dpo_dataset(args, tokenizer, dataset_dict, percentage = 1):
    if dataset_dict[args.dataset_name] == "dpo-mix-7k-simplified":
        dataset = load_dataset(
            args.dataset_name, 
            split=[f"train[:{percentage}%]", f"test[:{percentage}%]"]
        )
        dataset = DatasetDict({"train": dataset[0], "test": dataset[1]})
    return dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import argparse
    from termcolor import colored

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/deita-10k-v0-sft")
    parser.add_argument("--data_percentage", type=int, default=1)
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()

    dataset_dict = {
        "HuggingFaceH4/deita-10k-v0-sft": "deita-10k-v0-sft",
        "alvarobartt/dpo-mix-7k-simplified": "dpo-mix-7k-simplified",
    }

    sft_dataset = load_sft_dataset(
        args,
        dataset_dict,
        percentage = 1
    )
    print(f"{colored('Loaded SFT dataset', 'cyan')}: {sft_dataset}")

    args.dataset_name = "alvarobartt/dpo-mix-7k-simplified"
    tokenizer = AutoTokenizer.from_pretrained("unsloth/tinyllama-bnb-4bit")
    dpo_dataset = load_dpo_dataset(
        args,
        tokenizer,
        dataset_dict,
        percentage = 1
    )
    print(f"{colored('Loaded DPO dataset', 'cyan')}: {dpo_dataset}")
