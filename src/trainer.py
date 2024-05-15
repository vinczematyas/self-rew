import torch
from trl import SFTTrainer, DPOTrainer
from transformers import TrainingArguments
from unsloth import PatchDPOTrainer
PatchDPOTrainer()

def get_sft_trainer(args, model, tokenizer, dataset):
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
        seed = args.seed,
        output_dir = args.output_dir,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        max_seq_length = args.max_seq_length,
        args = training_args
    )

    return trainer

def get_dpi_trainer(args, model, tokenizer, dataset):
    training_args = TrainingArguments(
        do_eval=True,
        evaluation_strategy = "steps",
        eval_steps = 100,
        save_strategy = "epoch",
        per_device_train_batch_size = 1, #Zephyr
        gradient_accumulation_steps = 16, #Zephyr
        per_device_eval_batch_size = 1,
        warmup_ratio = 0.1, #Zephyr
        num_train_epochs = 2, #Zephyr
        learning_rate = 5.0e-07, #Zephyr
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 100,
        optim = "paged_adamw_8bit",
        lr_scheduler_type = "cosine", #Zephyr
        seed = args.seed,
        output_dir = args.output_dir,
    )

    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.05, #Zephyr
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer
    )

    return trainer
