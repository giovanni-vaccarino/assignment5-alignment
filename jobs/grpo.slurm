#!/bin/bash
# One GRPO run per job: policy on cuda:0, vLLM (rollouts + eval) on cuda:1 (same node).
# Usage (from repo root):  sbatch -J <run_name> jobs/grpo.slurm --run-name <run_name> [grpo.py flags...]
# gpuA100x4-interactive: 1 h max, QOS allows 2 submitted / 1 running per user. Size runs (n-grpo-steps) to fit.
#SBATCH --account=bhag-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Default HF_HOME (/work/hdd/bghp/...) is not readable; weights are pre-downloaded here (no internet on compute nodes).
export HF_HOME=/projects/bhag/gvaccarino/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Reduce fragmentation OOMs on 40 GB A100s (grpo_base 22550804 OOMed with 2.9 GB reserved-but-unallocated)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
start=$(date +%s)
uv run --no-sync python -m cs336_alignment.grpo.grpo "$@"
echo "end=$(date) wall_s=$(( $(date +%s) - start ))"
