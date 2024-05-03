from transformers import TextStreamer
from unsloth import FastLanguageModel

from src.model import load_model

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

model, tokenizer = load_model("unsloth/tinyllama-bnb-4bit")
FastLanguageModel.for_inference(model)  # 2x faster inference

inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Continue the fibonnaci sequence.",  # instruction
            "1, 1, 2, 3, 5, 8",  # input
            "",  # output
        )
    ], return_tensors="pt").to(model.device)

_ = model.generate(**inputs, streamer=TextStreamer(tokenizer), max_tokens=256)

