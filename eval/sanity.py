"""
Check response of models

python ./eval/sanity.py --model=./runs/qwen3-openorca/final

"""

import re
import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Registers model
from models.hybrid_qwen import HyperbolicQwen


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant answering multiple-choice science questions. "
    "Always end your response with 'Answer: X' where X is the single letter of "
    "the correct choice (A, B, C, or D)."
)

def format_question(example: dict) -> str:
    """Format a single ARC example into a prompt string."""
    question = example["question"]
    choices = example["choices"]          # {"label": [...], "text": [...]}
    labels = choices["label"]
    texts  = choices["text"]

    choice_str = "\n".join(
        f"{label}. {text}" for label, text in zip(labels, texts)
    )
    return f"Question: {question}\n{choice_str}\n"


def build_fewshot_prompt(fewshot_examples: list[dict], test_example: dict) -> str:
    """
    Build a full prompt with k few-shot examples followed by the test question.
    Each few-shot example shows the answer so the model learns the format.
    """
    prompt = SYSTEM_PROMPT + "\n\n"
    for ex in fewshot_examples:
        prompt += format_question(ex)
        prompt += f"Answer: {ex['answerKey']}\n\n"
    prompt += format_question(test_example)
    # Leave the answer prefix for the model to complete
    prompt += "Answer:"
    return prompt


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> Optional[str]:
    """
    Extract the predicted answer letter from the model's response.
    Tries several patterns in order of specificity.
    """
    # Pattern 1: explicit "Answer: X" (our prompted format)
    match = re.search(r"Answer:\s*([A-Da-d])", response)
    if match:
        return match.group(1).upper()

    # Pattern 2: standalone letter on its own line, e.g. "\nA\n"
    match = re.search(r"^\s*([A-Da-d])\s*$", response, re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Pattern 3: letter followed by period or parenthesis, e.g. "A." or "(A)"
    match = re.search(r"\b([A-Da-d])[.)]", response)
    if match:
        return match.group(1).upper()

    return None  # failed to extract


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 16,
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a causal LM on ARC-Challenge.")
    parser.add_argument("--model",    type=str, required=True,  help="HuggingFace model name or local path")
    parser.add_argument("--tokenizer",type=str, default=None,   help="Huggingface model name (if shared by model), defaults to model")
    parser.add_argument("--sample",   type=int, default=100,    help="Number of test questions to evaluate on")
    parser.add_argument("--fewshot",  type=int, default=5,      help="Number of few-shot examples per question")
    parser.add_argument("--device",   type=str, default="cuda", help="Device: 'cuda', 'cpu', or 'mps'")
    parser.add_argument("--seed",     type=int, default=42,     help="Random seed")
    parser.add_argument("--verbose",  action="store_true",      help="Print per-example predictions")
    parser.add_argument("--dtype",    type=str, default="auto", help="Model dtype: 'auto', 'float16', 'bfloat16'")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Number of tokens to generate")
    parser.add_argument("--prompt", type=str, default="The sky is", help="Model prompt")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model} with device {args.device}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
    torch_dtype = dtype_map.get(args.dtype, "auto")

    if args.tokenizer is None:
        tokenizer_path = args.model
    else:
        tokenizer_path = args.tokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device,
    )
    model.eval()

    # prompt = "What is 7+8? Please answer succintly:"
    prompt = args.prompt

    response = generate_answer(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, device=args.device)

    print("\n=== Prompt ===")
    print(prompt)
    print("\n=== Response ===")
    print(response)
    print()


if __name__ == "__main__":
    main()