import torch
from trl import SFTTrainer
from transformers import TrainingArguments

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
        train_dataset = dataset[0],
        eval_dataset = dataset[1],
        max_seq_length = args.max_seq_length,
        args = training_args
    )

    return trainer

