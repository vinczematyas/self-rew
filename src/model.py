from unsloth import FastLanguageModel

def load_model(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    return model, tokenizer

def get_peft_model(args, model):
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = args.lora_alpha,
        use_gradient_checkpointing = True, # IF YOU GET OUT OF MEMORY
        random_state = args.seed,
        loftq_config = None, # LoftQ
        max_seq_length = 2048,
    )
    return model
