#!/bin/bash
# Submit the full experiment pipeline to SLURM.
# Each model runs as a separate job; steering depends on extraction.
#
# Usage:
#   bash slurm/run_all.sh
#   MODEL=llama3.1-8b bash slurm/run_all.sh   # single model

set -euo pipefail

LOG_DIR="/lustre/nvwulf/scratch/nijjohnson/logs"
mkdir -p "$LOG_DIR"

MODELS=("llama3.1-8b" "gemma2-9b" "qwen2.5-7b")

# If MODEL env var is set, only run that model
if [[ -n "${MODEL:-}" ]]; then
    MODELS=("$MODEL")
fi

for model in "${MODELS[@]}"; do
    echo "=== Submitting jobs for $model ==="

    # Phase 1: Extraction
    EXTRACT_JOB=$(sbatch --export=ALL,MODEL="$model" --parsable slurm/extract_vectors.sbatch)
    echo "  Extraction: $EXTRACT_JOB"

    # Phase 2: Baselines (can run in parallel with extraction)
    BASELINE_JOB=$(sbatch --export=ALL,MODEL="$model" --parsable slurm/run_baselines.sbatch)
    echo "  Baselines:  $BASELINE_JOB"

    # Phase 3: Steering (depends on extraction)
    STEER_JOB=$(sbatch --export=ALL,MODEL="$model" --dependency=afterok:$EXTRACT_JOB --parsable slurm/run_steering.sbatch)
    echo "  Steering:   $STEER_JOB (depends on $EXTRACT_JOB)"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
