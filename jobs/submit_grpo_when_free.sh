#!/bin/bash
# Submit ONE grpo run as soon as the gpuA100x4-interactive QOS allows it (max 1 submitted job per user as of 2026-10-01).
# Retries every 60 s until sbatch accepts. Run detached so it survives the session:
#   setsid nohup jobs/submit_grpo_when_free.sh <run_name> [grpo.py flags...] >> outputs/slurm/submitter.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
name="$1"; shift
# Guard: refuse to start if another submitter for the same run is alive, or the run dir already exists.
if [ -e "outputs/grpo/$name" ] || [ "$(pgrep -u "$USER" -fc "submit_grpo_when_free.sh $name ")" -gt 1 ]; then echo "$(date) $name: refusing (run dir exists or another submitter is alive)"; exit 1; fi
until out=$(sbatch -J "$name" jobs/grpo.slurm --run-name "$name" "$@" 2>&1); do
  sleep 60
done
echo "$(date) $name: $out"
