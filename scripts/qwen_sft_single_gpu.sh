#!/bin/bash

#SBATCH --job-name=Qwen3-1.7B-SFT
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --mail-type=ALL
#SBATCH --output=logs/slurm/o_%A.out
#SBATCH --error=logs/slurm/o_%A.err
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --requeue

# Example wrapper, adjust to your system.

# you have to creat log folder first
# mkdir -p log/slurm/

date
hostname
pwd

# torch==2.9.0+cu129
source ../hyperbolic_rag/vllm_rag_env/bin/activate
ml load CUDA/12.1.1
ml load GCC/10.2.0

nvidia-smi

# Start the training
# python train_qwen3_openorca.py

# If there is an error, print out the error code
echo $?

echo "All done in sbatch."
date