from datasets import load_dataset, DatasetDict

def load_sft_dataset(args, percentage = 1):
    if args.dataset == "HuggingFaceH4/deita-10k-v0-sft":
        dataset = load_dataset(
            args.dataset,
            split=[f"train_sft[:{percentage}%]", f"test_sft[:{percentage}%]"]
        )
        dataset = dataset[0]  # train only
    elif args.dataset == "habanoz/lima-chat-format":
        dataset = load_dataset(
            args.dataset, 
            split=f"train[:{percentage}%]"
        )
        def preprocess_data(example):
            example["category"] = example["conversation"][0]["content"]
            example["conversation"] = example["conversation"][1]["content"]
            return example
        dataset = dataset.map(preprocess_data)
        dataset = dataset.rename_column("category", "prompt")
        dataset = dataset.rename_column("conversation", "response")
    else:
        dataset = load_dataset("parquet", data_files=args.dataset, split=f"train[:{percentage}%]")
    return dataset


def load_dpo_dataset(args, tokenizer, dataset_dict, percentage = 1):
    if args.dataset == "alvarobartt/dpo-mix-7k-simplified":
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

