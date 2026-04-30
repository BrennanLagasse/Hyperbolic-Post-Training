import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import json
from datetime import datetime

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

def save_results(
    model_name: str,
    model_path: str,
    dataset_name: str,
    dataset_path: str,
    metrics: dict,
    results_dir: str = "./results",
    extra_metadata: dict = None,
):
    """
    Save evaluation scores to a JSON file in results_dir.
    
    Args:
        model_name:     name of the model
        model_path:     path to model params
        dataset_name:   name of the dataset used
        matrics:        dict of metric_name -> value
        results_dir:    directory to write results to
        extra_metadata: any additional info to store (hyperparams, etc.)
    
    Example output file: ./results/hyperbolic-qwen_openwebtext_20260424_153012.json
    """

    os.makedirs(results_dir, exist_ok=True)

    # Build a clean model slug for the filename
    model_slug = os.path.basename(model_name.rstrip("/"))
    dataset_slug = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{model_slug}_{dataset_slug}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    record = {
        "model": model_path,
        "dataset": dataset_path,
        "timestamp": timestamp,
        "metrics": metrics,
        **(extra_metadata or {}),
    }

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Results saved to {filepath}")
    return filepath