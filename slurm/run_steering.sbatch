#!/bin/bash
#SBATCH --job-name=steer-gen
#SBATCH --partition=h200x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/steering_%j.out
#SBATCH --error=logs/steering_%j.err

# --- Environment Setup ---
module load cuda12.8/toolkit/12.8.1
conda activate steering-study

export PROJECT_DIR=/lustre/nvwulf/scratch/nijjohnson/activation-steering-study
export HF_HOME=/lustre/nvwulf/scratch/nijjohnson/hf_cache
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token)
export PYTHONPATH=$PROJECT_DIR

# --- Working Directory ---
cd $PROJECT_DIR
mkdir -p logs

# --- Run Steering ---
python scripts/run_steering.py \
    --config configs/default.yaml \
    ${MODEL:+--model $MODEL} \
    ${PERSONA:+--persona $PERSONA}

echo "Steering job $SLURM_JOB_ID completed at $(date)"
