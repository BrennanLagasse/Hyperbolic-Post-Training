"""
Supervised Fine-Tuning: Qwen3-1.7B on OpenOrca
================================================
Features:
  - Hugging Face Trainer + TRL SFTTrainer
  - W&B experiment tracking
  - Periodic checkpointing (resume-safe)
  - LoRA / QLoRA (4-bit) for memory efficiency
  - Gradient checkpointing
  - Configurable via dataclass or CLI overrides

Usage:
  # Full run
  python train_qwen3_openorca.py

  # Override any field via CLI (uses HfArgumentParser)
  python train_qwen3_openorca.py \
      --output_dir ./runs/my_run \
      --num_train_epochs 2 \
      --per_device_train_batch_size 4 \
      --use_4bit True

  # Resume from a checkpoint
  python train_qwen3_openorca.py --resume_from_checkpoint ./runs/qwen3-openorca/checkpoint-500
"""

# source ../hyperbolic_rag/vllm_rag_env/bin/activate

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, SFTConfig

#, DataCollatorForCompletionOnlyLM

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
        default="Open-Orca/OpenOrca",
        metadata={"help": "Huggingface dataset id"},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit dataset size for quick tests (None = use all)"},
    )
    # Sequence length (i.e. context length)
    max_length: int = field(
        default=1024,
        metadata={"help": "Maximum token length per example"},
    )
    # LoRA
    use_lora: bool = field(default=True, metadata={"help": "Enable LoRA"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of modules to apply LoRA to"},
    )
    # Quantization
    use_4bit: bool = field(default=True, metadata={"help": "Load model in 4-bit (QLoRA)"})
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit layers (bfloat16 / float16)"},
    )
    # W&B
    wandb_project: str = field(default="qwen3-openorca-sft", metadata={"help": "W&B project name"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "W&B run name"})
    # Output / checkpointing
    output_dir: str = field(default="./runs/qwen3-openorca-v2")
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint dir, or 'True' to auto-resume latest"},
    )
    seed: int = field(default=42)


# ---------------------------------------------------------------------------
# Prompt formatting
# See docs at https://huggingface.co/docs/transformers/en/chat_templating
# ---------------------------------------------------------------------------

SYSTEM_TOKEN = "<|im_start|>system\n"
USER_TOKEN = "<|im_start|>user\n"
ASSISTANT_TOKEN = "<|im_start|>assistant\n"
END_TOKEN = "<|im_end|>\n"

def format_example(example: dict) -> str:
    """ Convert from OpenOrca to ChatML formatting """
    
    system = example.get("system_prompt", "").strip()
    question = example.get("question", "").strip()
    response = example.get("response", "").strip()

    parts = []
    if system:
        parts.append(f"{SYSTEM_TOKEN}{system}{END_TOKEN}")
    parts.append(f"{USER_TOKEN}{question}{END_TOKEN}")
    parts.append(f"{ASSISTANT_TOKEN}{response}{END_TOKEN}")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser((ScriptArguments,))
    # Merge with a subset of TrainingArguments via CLI if needed
    (args,) = parser.parse_args_into_dataclasses()

    set_seed(args.seed)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.init(
        entity="brennan-lagasse-yale-university",
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        resume="allow",  # allows seamless resume if run_id is stored
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info(f"\n\nLoading dataset: {args.dataset_name} / {args.dataset_split}")
    
    raw_dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        streaming=True,
    )
    raw_dataset = raw_dataset.take(args.max_samples if args.max_samples else 23_000)

    # SFTTrainer needs a regular Dataset, not an IterableDataset
    from datasets import Dataset
    raw_dataset = Dataset.from_generator(lambda: (ex for ex in raw_dataset))

    logger.info(f"\n\nDataset size: {len(raw_dataset):,}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Quantization config ───────────────────────────────────────────────────
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
    logger.info(f"\n\nLoading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=None, # Needs to be set to None for distributed training
        torch_dtype=compute_dtype if not args.use_4bit else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    model.config.use_cache = False  # required for gradient checkpointing

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

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,

        # Batch size & accumulation  ← tune to your GPU VRAM
        per_device_train_batch_size=4,   # Number of training samples on a signle GPU per pass
        gradient_accumulation_steps=2,   # Number of mini-batches per gradient update

        # Sequence length (SFTConfig-specific)
        max_length=args.max_length,
        dataset_text_field="text",   # Name of the column that contains text data in the dataset
        packing=True,                # pack short examples → fewer padding tokens

        # Logging (not very well documented)
        logging_steps=10,
        report_to="wandb",

        # Checkpointing: saves every 500 steps; keeps last 3
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        seed=args.seed,
        dataloader_num_workers=4,
    )

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_dataset,
        formatting_func=format_example,   # converts dict → ChatML string
    )

    # ── Train (with optional resume) ──────────────────────────────────────────
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        # Auto-detect latest checkpoint in output_dir
        resume = True

    logger.info("\n\nStarting training…")
    trainer.train(resume_from_checkpoint=resume)

    # ── Save final artefacts ──────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Model saved to {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()