from unsloth import FastLanguageModel

def load_model(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    return model, tokenizer

def get_peft_model(model, lora_r = 8, lora_alpha = 32):
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        use_gradient_checkpointing = True, # IF YOU GET OUT OF MEMORY
        random_state = 42,
        loftq_config = None, # LoftQ
    )
    return model

