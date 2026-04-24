"""
Hyperbolic Head Pretraining: Hybrid Qwen3-1.7B on OpenWebText
================================================================
Features:
  - Freezes all weights except the vocab embedding projection head (w)
  - Trains only the hyperbolic projection head weights on OpenWebText
  - Hugging Face Trainer for distributed training across 2 GPUs
  - W&B experiment tracking
  - Periodic checkpointing (resume-safe)
  - Gradient checkpointing for memory efficiency
  - Configurable via dataclass or CLI overrides

Usage:
  # Full run (single GPU)
  python train_hyperbolic_head.py

  # Multi-GPU (2 GPUs) via torchrun
  torchrun --nproc_per_node=2 train_hyperbolic_head.py

  # Override any field via CLI
  torchrun --nproc_per_node=2 train_hyperbolic_head.py \
      --output_dir ./runs/my_run \
      --num_train_epochs 2 \
      --per_device_train_batch_size 4

  # Resume from a checkpoint
  torchrun --nproc_per_node=2 train_hyperbolic_head.py \
      --resume_from_checkpoint ./runs/hyperbolic-head/checkpoint-500
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import wandb
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

from models.hybrid_qwen import HyperbolicQwen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    # Model
    model_name: str = field(
        default="./hyperbolic-qwen",
        metadata={"help": "Path to the model being trained"},
    )
    # Tokenizer
    tokenizer_name: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "HF model id or local path of the model with the tokenizer"},
    )
    # Dataset
    dataset_name: str = field(
        default="Skylion007/openwebtext",
        metadata={"help": "HuggingFace dataset id"},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit dataset size for quick tests (None = use all)"},
    )
    # Sequence length
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum token length per example"},
    )
    # W&B
    wandb_project: str = field(
        default="hyperbolic-head-openwebtext",
        metadata={"help": "W&B project name"},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "W&B run name"},
    )
    # Output / checkpointing
    output_dir: str = field(default="./runs/hyperbolic-head")
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint dir, or 'True' to auto-resume latest"},
    )
    seed: int = field(default=42)


# ---------------------------------------------------------------------------
# Model wrapper: frozen backbone + trainable hyperbolic head
# ---------------------------------------------------------------------------

class HyperbolicHeadModel(nn.Module):
    """
    Wraps HybridQwen so that only the hyperbolic projection head weights (w)
    are trainable. All other parameters are frozen.

    Assumes HybridQwen exposes:
      - self.hyperbolic_head.w  : (vocab_size, emb_dim) projection weights
      - forward(input_ids, attention_mask, labels) -> CausalLMOutput or loss
    """

    def __init__(self, model_name: str, compute_dtype: torch.dtype):
        super().__init__()
        self.model = HyperbolicQwen.from_pretrained(
            model_name,
            # torch_dtype=compute_dtype,
            # trust_remote_code=True,
            # attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        )
        self.model.config.use_cache = False  # required for gradient checkpointing
        self._freeze_all_except_head()

    def _freeze_all_except_head(self):
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only the hyperbolic head weights
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    # Expose config so Trainer can inspect it
    @property
    def config(self):
        return self.model.config

    # Gradient checkpointing functionality from nn.Module
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, model_name: str, compute_dtype=torch.bfloat16):
        instance = cls(model_name, compute_dtype)
        return instance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser((ScriptArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    set_seed(args.seed)

    # ── W&B ──────────────────────────────────────────────────────────────────
    # Only initialise on the main process to avoid duplicate runs
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity="brennan-lagasse-yale-university",
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            resume="allow",
        )

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info(f"\n\nLoading dataset: {args.dataset_name} / {args.dataset_split}")

    raw_dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        streaming=True,
        trust_remote_code=True,
    )
    raw_dataset = raw_dataset.take(args.max_samples if args.max_samples else 100_000)
    raw_dataset = Dataset.from_generator(lambda: (ex for ex in raw_dataset))

    logger.info(f"\n\nDataset size: {len(raw_dataset):,}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"\n\nTokenizing Dataset")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,   # DataCollator handles padding
        )

    tokenized_dataset = raw_dataset.map(
        tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    # Causal LM collator: shifts labels automatically, pads batches
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,   # causal LM, not masked
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"\n\nLoading model: {args.model_name}")
    compute_dtype = torch.bfloat16
    model = HyperbolicHeadModel(args.model_name, compute_dtype)

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,

        # Batch size & accumulation  ← tune to your GPU VRAM
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,   # effective batch = 4 * 4 * 2 GPUs = 32

        # Memory
        gradient_checkpointing=True,
        bf16=True,
        dataloader_num_workers=4,

        # Multi-GPU: DDP is used automatically by torchrun
        ddp_find_unused_parameters=False,  # all fwd params are used; avoids DDP overhead

        # Logging
        logging_steps=10,
        report_to="wandb" if os.environ.get("LOCAL_RANK", "0") == "0" else "none",

        # Checkpointing: saves every 500 steps; keeps last 3
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        seed=args.seed,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        # tokenizer=tokenizer,
    )

    # ── Train (with optional resume) ──────────────────────────────────────────
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True

    logger.info("\n\nStarting training…")
    trainer.train(resume_from_checkpoint=resume)

    # ── Save final artefacts ──────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    # trainer.save_model(final_dir)

    # Save the base model (not the wrapper)
    model.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Model saved to {final_dir}")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.finish()


if __name__ == "__main__":
    main()