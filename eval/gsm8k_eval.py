"""
GSM8K evaluation script using chain-of-thought prompting and answer extraction.

The GSM8K dataset contains grade-school math problems. Ground truth answers
are integers, typically preceded by "####" in the answer field.

Usage:
    python eval_gsm8k.py --model <model_name_or_path> [--sample 100] [--fewshot 8] [--device cuda]

Example:
    python eval_gsm8k.py --model meta-llama/Llama-3.2-1B-Instruct --sample 100
"""

import re
import argparse
import random
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from eval_utils import generate_answer


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def extract_gold_answer(answer_str: str) -> Optional[str]:
    """
    Extract the integer answer from the GSM8K ground truth string.
    Ground truth answers are formatted like:
        "... some reasoning ... #### 42"
    We extract the number after ####, stripping commas (e.g. "1,000" -> "1000").
    """
    match = re.search(r"####\s*([\d,\-]+)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last integer in the string
    matches = re.findall(r"-?\d+(?:,\d{3})*", answer_str)
    if matches:
        return matches[-1].replace(",", "")
    return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant solving grade-school math problems step by step. "
    "Show your reasoning, then end your response with '#### <integer>' where "
    "<integer> is the final numerical answer."
)


def format_problem(example: dict) -> str:
    return f"Problem: {example['question']}\nSolution:"


def build_fewshot_prompt(fewshot_examples: list[dict], test_example: dict) -> str:
    """
    Build a chain-of-thought few-shot prompt. Each few-shot example shows
    the full reasoning and the final #### answer, teaching the model the format.
    """
    prompt = SYSTEM_PROMPT + "\n\n"
    for ex in fewshot_examples:
        gold = extract_gold_answer(ex["answer"])
        # Use the dataset's own solution as the CoT demonstration, but
        # re-format the final line to ensure consistent #### format
        solution_body = re.sub(r"####.*", "", ex["answer"]).strip()
        prompt += format_problem(ex) + "\n"
        prompt += solution_body + f"\n#### {gold}\n\n"
    prompt += format_problem(test_example) + "\n"
    return prompt


# ---------------------------------------------------------------------------
# Answer extraction from model response
# ---------------------------------------------------------------------------

def extract_predicted_answer(response: str) -> Optional[str]:
    """
    Extract the predicted integer from the model's response.
    Tries patterns in order of specificity.
    """
    # Pattern 1: explicit "#### 42" (our prompted format)
    match = re.search(r"####\s*([\d,\-]+)", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Pattern 2: "the answer is 42" / "= 42" at end of response
    match = re.search(
        r"(?:answer is|answer:|therefore|total is|=)\s*\$?\s*(-?[\d,]+)\s*\.?\s*$",
        response, re.IGNORECASE
    )
    if match:
        return match.group(1).replace(",", "").strip()

    # Pattern 3: last integer in the response
    matches = re.findall(r"-?\d+(?:,\d{3})*", response)
    if matches:
        return matches[-1].replace(",", "")

    return None


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Normalize to a canonical integer string for comparison."""
    if answer is None:
        return None
    try:
        # Handle floats that are actually integers e.g. "42.0"
        return str(int(float(answer.replace(",", ""))))
    except ValueError:
        return answer.strip()

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    total: int
    correct: int
    failed_extraction: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Accuracy:            {self.accuracy:.2%}  ({self.correct}/{self.total})\n"
            f"Extraction failures: {self.failed_extraction}/{self.total}"
        )


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample_size: int = 100,
    num_fewshot: int = 8,
    seed: int = 42,
    device: str = "cuda",
    verbose: bool = False,
) -> EvalResult:
    random.seed(seed)

    dataset = load_dataset("openai/gsm8k", "main")
    train_split = list(dataset["train"])
    test_split  = list(dataset["test"])

    sample = random.sample(test_split, min(sample_size, len(test_split)))

    correct = 0
    failed = 0

    for example in tqdm(sample, desc="Evaluating"):
        fewshot_examples = random.sample(train_split, min(num_fewshot, len(train_split)))

        prompt = build_fewshot_prompt(fewshot_examples, example)
        response = generate_answer(model, tokenizer, prompt, device=device, max_new_tokens=512)

        predicted = normalize_answer(extract_predicted_answer(response))
        gold      = normalize_answer(extract_gold_answer(example["answer"]))

        if predicted is None:
            failed += 1
            if verbose:
                print(f"[EXTRACTION FAILED] response='{response[:80]}...' gold={gold}")
        elif predicted == gold:
            correct += 1
            if verbose:
                print(f"[CORRECT] predicted={predicted} gold={gold}")
        else:
            if verbose:
                print(f"[WRONG]   predicted={predicted} gold={gold}")
                print(f"          response='{response[:120]}...'")

    return EvalResult(total=len(sample), correct=correct, failed_extraction=failed)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a causal LM on GSM8K.")
    parser.add_argument("--model",   type=str, required=True,   help="HuggingFace model name or local path")
    parser.add_argument("--sample",  type=int, default=100,     help="Number of test questions to evaluate on")
    parser.add_argument("--fewshot", type=int, default=8,       help="Number of few-shot CoT examples per question")
    parser.add_argument("--device",  type=str, default="cuda",  help="Device: 'cuda', 'cpu', or 'mps'")
    parser.add_argument("--seed",    type=int, default=42,      help="Random seed")
    parser.add_argument("--verbose", action="store_true",       help="Print per-example predictions")
    parser.add_argument("--dtype",   type=str, default="auto",  help="Model dtype: 'auto', 'float16', 'bfloat16'")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
    torch_dtype = dtype_map.get(args.dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device,
    )
    model.eval()

    print(f"Evaluating on {args.sample} GSM8K examples with {args.fewshot}-shot CoT prompting...\n")
    result = evaluate(
        model=model,
        tokenizer=tokenizer,
        sample_size=args.sample,
        num_fewshot=args.fewshot,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
    )

    print("\n=== Results ===")
    print(result)


if __name__ == "__main__":
    main()