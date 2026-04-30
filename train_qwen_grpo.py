"""
GRPO Fine-Tuning: Qwen3-1.7B (Base/Hybrid) on OpenMathInstruct-2
===========================================================
Features:
  - TRL GRPOTrainer with group relative policy optimization
  - Math-specific reward functions (correctness + format)
  - W&B experiment tracking
  - LoRA / QLoRA (4-bit) for memory efficiency
  - Periodic checkpointing (resume-safe)
  - Configurable via dataclass or CLI overrides

Usage:
  # Full run
  python train_qwen3_grpo_math.py

  # Multi-GPU
  torchrun --nproc_per_node=4 train_qwen3_grpo_math.py

  # Override any field via CLI
  python train_qwen3_grpo_math.py \
      --output_dir ./runs/grpo_math \
      --num_train_epochs 2 \
      --per_device_train_batch_size 4

  # Resume from checkpoint
  python train_qwen3_grpo_math.py \
      --resume_from_checkpoint ./runs/grpo_math/checkpoint-500
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
from trl import GRPOTrainer, GRPOConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    # Model
    model_name: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "HF model id or local path"},
    )
    # Dataset
    dataset_name: str = field(
        default="nvidia/OpenMathInstruct-2",
        metadata={"help": "HuggingFace dataset id"},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"},
    )
    max_samples: Optional[int] = field(
        default=50_000,
        metadata={"help": "Limit dataset size for quick tests (None = use all)"},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "Maximum token length for input prompt"},
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum tokens to generate per response"},
    )
    # GRPO-specific
    num_generations: int = field(
        default=8,
        metadata={"help": "Number of completions to sample per prompt (group size)"},
    )
    # LoRA
    use_lora: bool = field(default=True, metadata={"help": "Enable LoRA"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated LoRA target modules"},
    )
    # Quantization
    use_4bit: bool = field(default=True, metadata={"help": "Load model in 4-bit (QLoRA)"})
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit layers"},
    )
    # W&B
    wandb_project: str = field(default="qwen3-grpo-math", metadata={"help": "W&B project"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "W&B run name"})
    # Output
    output_dir: str = field(default="./runs/grpo_math")
    resume_from_checkpoint: Optional[str] = field(default=None)
    seed: int = field(default=42)


# ---------------------------------------------------------------------------
# Dataset formatting
# OpenMathInstruct-2 schema:
#   - problem:           the math problem string
#   - generated_solution: full solution including answer
#   - expected_answer:   ground truth answer string
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, then state your final answer "
    "inside \\boxed{}."
)

def format_prompt(example: dict) -> dict:
    """
    Convert OpenMathInstruct-2 example into a prompt/answer pair.
    GRPOTrainer expects:
      - 'prompt': the input to generate from (list of chat messages)
      - 'answer': ground truth used by reward functions
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": example["problem"]},
        ],
        "answer": example["expected_answer"],
    }


# ---------------------------------------------------------------------------
# Reward functions
# GRPOTrainer calls each reward_func(prompts, completions, **kwargs) -> list[float]
# 'completions' is a list of generated strings (one per prompt in the batch)
# Any extra dataset columns are passed as kwargs (e.g. 'answer')
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> Optional[str]:
    """Pull the content of the last \\boxed{} in a string."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans: str) -> str:
    """Light normalization for answer comparison."""
    ans = ans.strip().lower()
    # Remove trailing punctuation and whitespace
    ans = re.sub(r"[,\.\s]+$", "", ans)
    # Remove thousand separators
    ans = ans.replace(",", "")
    return ans


def reward_correctness(prompts, completions, answer, **kwargs) -> list[float]:
    """
    +1.0  if the boxed answer matches the ground truth exactly
     0.0  if boxed answer is present but wrong
    -0.5  if no boxed answer found at all
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        predicted = extract_boxed_answer(completion)
        if predicted is None:
            rewards.append(-0.5)
        elif normalize_answer(predicted) == normalize_answer(str(gt)):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_format(prompts, completions, **kwargs) -> list[float]:
    """
    Reward well-structured responses:
      +0.2  for containing \\boxed{}
      +0.1  for having a reasoning step before the box
      -0.2  for being very short (likely no reasoning)
    """
    rewards = []
    for completion in completions:
        score = 0.0
        has_box = bool(re.search(r"\\boxed\{", completion))
        has_reasoning = len(completion.strip()) > 100
        is_too_short = len(completion.strip()) < 20

        if has_box:
            score += 0.2
        if has_reasoning:
            score += 0.1
        if is_too_short:
            score -= 0.2

        rewards.append(score)
    return rewards


def reward_no_repetition(prompts, completions, **kwargs) -> list[float]:
    """
    Small penalty for repetitive outputs, which can indicate degenerate generations.
    Checks for repeated consecutive sentences.
    """
    rewards = []
    for completion in completions:
        sentences = re.split(r"[.!?]+", completion)
        sentences = [s.strip() for s in sentences if s.strip()]
        unique = len(set(sentences))
        total = len(sentences)
        repetition_ratio = 1.0 - (unique / total) if total > 0 else 0.0
        rewards.append(-0.3 * repetition_ratio)
    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser((ScriptArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    set_seed(args.seed)

    # ── W&B ──────────────────────────────────────────────────────────────────
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity="brennan-lagasse-yale-university",
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            resume="allow",
        )

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset_name} / {args.dataset_split}")

    raw_dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        streaming=True,
        trust_remote_code=True,
    )
    raw_dataset = raw_dataset.take(args.max_samples if args.max_samples else 50_000)
    raw_dataset = Dataset.from_generator(lambda: (ex for ex in raw_dataset))

    logger.info(f"Dataset size: {len(raw_dataset):,}")

    # Format into prompt/answer pairs
    dataset = raw_dataset.map(
        format_prompt,
        remove_columns=raw_dataset.column_names,
        desc="Formatting prompts",
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="left"  # left-pad for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Quantization ──────────────────────────────────────────────────────────
    compute_dtype = (
        torch.bfloat16 if args.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    )

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=None,
        torch_dtype=compute_dtype if not args.use_4bit else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    if args.use_lora:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ── GRPO config ───────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,

        # Batch size — note: effective batch is larger because of num_generations
        # Each prompt generates num_generations completions before a gradient step
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,

        # Generation
        num_generations=args.num_generations,   # group size G in GRPO
        max_completion_length=args.max_new_tokens,

        # Optimization
        learning_rate=1e-5,
        max_grad_norm=0.1,          # important for stability in RL training
        warmup_steps=50,
        bf16=True,

        # GRPO-specific
        beta=0.01,                  # KL penalty coefficient vs reference model
                                    # higher = stay closer to base model

        # Logging
        logging_steps=10,
        report_to="wandb" if os.environ.get("LOCAL_RANK", "0") == "0" else "none",

        # Checkpointing
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        seed=args.seed,
        dataloader_num_workers=2,
    )

    # ── GRPOTrainer ───────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[
            reward_correctness,     # primary signal: is the answer right?
            reward_format,          # secondary: is the format good?
            reward_no_repetition,   # tertiary: penalize degenerate outputs
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True

    logger.info("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=resume)

    # ── Save ──────────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Model saved to {final_dir}")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.finish()


if __name__ == "__main__":
    main()