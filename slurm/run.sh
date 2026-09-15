#!/usr/bin/env bash
# Submit with site-specific resource flags, for example:
# sbatch --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=2-00:00:00 slurm/run.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python run_pipeline.py "$@"
