import torch
from trl import SFTTrainer
from transformers import TrainingArguments

def get_sft_trainer(model, tokenizer, ds, output_dir, seed):
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        optim = "adamw_8bit",
        learning_rate = 2e-5,
        lr_scheduler_type = "linear",
        weight_decay = 0.1,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = output_dir,
        seed = seed,
        logging_strategy="epoch", 
        save_strategy="epoch"
    )
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = ds,
        dataset_text_field = "text",
        max_seq_length = 2048,
        packing = False,
        args = training_args,
    )
    return trainer

