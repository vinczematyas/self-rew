from datasets import load_dataset

def load_sft_dataset(args, dataset_dict, percentage = 1):
    if dataset_dict[args.dataset_name] == "deita-10k-v0-sft":
        dataset = load_dataset(
            args.dataset_name, 
            split=[f"train_sft[:{percentage}%]", f"test_sft[:{percentage}%]"]
        )
        dataset = DatasetDict({"train": dataset[0], "test": dataset[1]})
    return dataset

def load_dpo_dataset(args, dataset_dict, percentage = 1):
    if dataset_dict[args.dataset_name] == "dpo-mix-7k":
        dataset = load_dataset(
            args.dataset_name, 
            split=[f"train[:{percentage}%]", f"test[:{percentage}%]"]
        )
        dataset = DatasetDict({"train": dataset[0], "test": dataset[1]})

        column_names = list(dataset["train"].features)

        def apply_dpo_template(example):
            if all(k in example.keys() for k in ("chosen", "rejected")):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            prompt_messages = example["chosen"][:-1]
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        return example

        dataset = dataset.map(
            apply_dpo_template,
            remove_columns=column_names,
            desc="Formatting comparisons with prompt template",
        )
        for split in ["train", "test"]:
            dataset[split] = dataset[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
            )

        return dataset

