import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.inference_mode()
def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda",
) -> str:
    """Run greedy decoding and return only the newly generated tokens as text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,       # greedy — deterministic and sufficient for extraction
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()