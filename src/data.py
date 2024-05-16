from datasets import load_dataset, DatasetDict

from src.utils import dataset_dict

def load_sft_dataset(args, dataset_dict, percentage = 1):
    if dataset_dict[args.dataset_name] == "deita-10k-v0-sft":
        dataset = load_dataset(
            args.dataset_name, 
            split=[f"train_sft[:{percentage}%]", f"test_sft[:{percentage}%]"]
        )
        dataset = DatasetDict({"train": dataset[0], "test": dataset[1]})
    elif dataset_dict[args.dataset_name] == "lima-chat":
        # SFT dataset from the paper LIMA: Less Is More for Alignment \
        # transformed to work with hugging face chat templates
        dataset = load_dataset(
            args.dataset_name, 
            split=[f"train[:{percentage}%]"]
        )[0]

        def preprocess_data(example):
            example["category"] = example["conversation"][0]["content"]
            example["conversation"] = example["conversation"][1]["content"]
            return example
        dataset = dataset.map(preprocess_data)
        dataset = dataset.rename_column("category", "prompt")
        dataset = dataset.rename_column("conversation", "response")
    return dataset

def load_dpo_dataset(args, tokenizer, dataset_dict, percentage = 1):
    if dataset_dict[args.dataset_name] == "dpo-mix-7k-simplified":
        dataset = load_dataset(
            args.dataset_name, 
            split=[f"train[:{percentage}%]", f"test[:{percentage}%]"]
        )
        dataset = DatasetDict({"train": dataset[0], "test": dataset[1]})

        def preprocess_data(example):
            example["prompt"] = example["prompt"][0]["content"]
            example["chosen"] = example["chosen"][0]["content"]
            example["rejected"] = example["rejected"][0]["content"]
            return example
        dataset = dataset.map(preprocess_data, remove_columns="dataset")

    return dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import argparse
    from termcolor import colored

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="habanoz/lima-chat-format")
    parser.add_argument("--data_percentage", type=int, default=1)
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()

    sft_dataset = load_sft_dataset(
        args,
        dataset_dict,
        percentage = 1
    )
    print(f"{colored('Loaded SFT dataset', 'cyan')}: {sft_dataset}")

