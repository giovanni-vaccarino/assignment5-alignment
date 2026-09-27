#!/bin/bash
# Submit queued sft runs one at a time as gpuA100x4-interactive slots free up (QOS: max 2 jobs/user).
# Usage: nohup jobs/submit_when_free.sh "name n_examples n_epochs" ... &
cd "$(dirname "$0")/.."
C="--lr 2e-5 --batch-size 32 --attn-implementation flash_attention_2 --n-eval-examples 1319 --seed 0 --eval-every 10"
for spec in "$@"; do
  set -- $spec
  until out=$(sbatch -J "$1" -t 00:40:00 jobs/sft.slurm --run-name "$1" --n-examples "$2" --n-epochs "$3" $C 2>&1); do
    sleep 60
  done
  echo "$(date) $1: $out"
done
