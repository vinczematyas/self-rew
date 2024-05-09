from datasets import load_dataset

def load_alpaca_sft(tokenizer):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]

        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)

        return {
            "text" : texts,
        }

    dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True)
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
    dataset = dataset.map(chatml_format, remove_columns=dataset.column_names)
    return dataset
