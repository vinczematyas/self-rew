import os
import argparse
import pandas as pd
from termcolor import colored
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel, SFTTrainer


parser = argparse.ArgumentParser(description="Phase 0: Supervised Fine-Tuning.")
parser.add_argument("-m", "--model_name", type=str, default="unsloth/tinyllama-bnb-4bit")
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--seed", type=int, default=420)
parser.add_argument("--max_seq_length", type=int, default=2048)
args = parser.parse_args()

model_dict = {
    "unsloth/tinyllama-bnb-4bit": "tinyllama"
}

args.output_dir = f"models/{model_dict[args.model_name]}/{args.run_name}/sft"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = args.max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = args.seed,
)

dataset = load_dataset("HuggingFaceH4/deita-10k-v0-sft", split=['train_sft[:1%]', 'validation_sft[:1%]'])

training_args = TrainingArguments(
    do_eval=True,
    evaluation_strategy = "steps",
    eval_steps = 100,
    save_strategy = "epoch",
    per_device_train_batch_size = 4, #Zephyr
    gradient_accumulation_steps = 4, #Zephyr
    per_device_eval_batch_size = 4,
    warmup_ratio = 0.1, #Zephyr
    num_train_epochs = 3, #Zephyr
    learning_rate = 2.0e-05, #Zephyr
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 100,
    optim = "adamw_8bit",
    lr_scheduler_type = "cosine", #Zephyr
    seed = 3407,
    output_dir = args.output_dir,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset[0],
    eval_dataset = dataset[1],
    max_seq_length = args.max_seq_length,
    args = training_args
    )

trainer_stats = trainer.train()

trainer.model.save_pretrained(args.output_dir)

