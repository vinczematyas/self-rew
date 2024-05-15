from datasets import load_dataset

def load_sft_dataset(args, dataset_dict, percentage = 1):
    if dataset_dict[args.dataset_name] == "deita-10k-v0-sft":
        dataset = load_dataset(
            args.dataset_name, 
            split=[f"train_sft[:{percentage}%]", f"test_sft[:{percentage}%]"]
        )
    return dataset

def load_capybara_ift(tokenizer):
    def chatml_format(example):
        system = ""  # INFO: no system message in this dataset

        # get everything except the last message as input
        prompt = tokenizer.apply_chat_template(example["chosen"][:-1], tokenize=False, add_generation_prompt=True)

        # get the last assistant responses
        chosen = example["chosen"][-1]["content"] + "</s>" 
        rejected = example["rejected"][-1]["content"] + "</s>" 

        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    dataset = load_dataset("argilla/distilabel-capybara-dpo-7k-binarized", split = "train")
    print(dataset)
    dataset = dataset.filter(lambda r: r["messages"] < 2)
    dataset = dataset.map(chatml_format, remove_columns=dataset.column_names)
    return dataset
