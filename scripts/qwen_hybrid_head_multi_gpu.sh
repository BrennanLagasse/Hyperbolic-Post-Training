#!/bin/bash

#SBATCH --job-name=Qwen3-1.7B-Hybrid-Head-PT
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=logs/slurm/o_%A.out
#SBATCH --error=logs/slurm/o_%A.err
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --requeue

# Example wrapper, adjust to your system.

# you have to creat log folder first
# mkdir -p log/slurm/

date
hostname
pwd

source ../hyperbolic_rag/vllm_rag_env/bin/activate
ml load CUDA/12.1.1

nvidia-smi

# Train with two GPUs using pytorch elastic
PYTHONPATH=. torchrun --nproc_per_node=2 train_qwen_hybrid_projector.py \
    --model_name=./hyperbolic-qwen \
    --max_samples=100000

# If there is an error, print out the error code
echo $?

echo "All done in sbatch."
date