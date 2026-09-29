# GRPO experiments on GSM8K (Qwen2.5-Math-1.5B)

Personal interview-prep exploration based on CS336 A5 §7–8 (GRPO). Not a course submission.
Same working style as SFT ([SFT_EXPERIMENTS.md](SFT_EXPERIMENTS.md)): agree on the plan first, one solid run first,
then decide what to run next. Don't mass-queue.

## Setup

- Script: `cs336_alignment/grpo/grpo.py` (typer CLI, `--help`). Loop is complete; loss pieces pass all 14
  `tests/test_grpo.py` tests (checked 2026-09-29).
- Data: GSM8K instead of MATH. Train questions `data/gsm8k/train.jsonl` (7473; only question + final number are used,
  not the reference solutions), val `data/gsm8k/test.jsonl` (1319).
- Model: `Qwen/Qwen2.5-Math-1.5B` base (RL-Zero style, no SFT first), cached at `/projects/bhag/gvaccarino/hf_cache`.
- Layout: policy trains on `cuda:0`; vLLM on `cuda:1` does both the rollouts and the evals. The policy weights are
  copied into vLLM at the start of every GRPO step, so rollouts come from the current policy.
- Reward: `r1_zero_reward_fn` (1 if formatted AND correct, else 0). `question_only_reward_fn` for the prompt ablation.
- Eval: temperature 1.0, max 1024 tokens, fixed sampling seed. Handout says >= 1024 examples.
- Zero-shot baseline (r1_zero prompt, from SFT notes): **accuracy 2.5%, format rate 16.8%**.
- Reference points from SFT: full-data SFT 62.7%, n128 SFT 64.2% (same eval setup, T=1).

### Handout defaults (on-policy)
`n_grpo_steps 200, lr 1e-5, rollout_batch_size 256 (= 32 questions x group_size 8), train_batch_size 256,
epochs_per_rollout_batch 1, gradient_accumulation_steps 128 (microbatch 2), loss reinforce_with_baseline,
use_std_normalization True, temperature 1.0, max_tokens 1024, min_tokens 4, AdamW betas (0.9, 0.95), wd 0, clip 1.0`.
One optimizer step per rollout batch -> exactly on-policy.

### Environment (NCSA Delta): same as SFT
- venv `.venv` -> `/work/nvme/bhag/gvaccarino/a5/venv` (never `/work/hdd`). `HF_HOME` overridden in the job script.
- SLURM: `jobs/grpo.slurm` (account `bhag-delta-gpu`, `gpuA100x4-interactive`, 2 GPUs, 16 CPU, 64G, **1 h max**;
  QOS: 2 submitted / 1 running per user). Logs in `outputs/slurm/`, runs in `outputs/grpo/<run>/`.
  ```
  sbatch -J <run> jobs/grpo.slurm --run-name <run> [flags...]
  ```
- `jobs/submit_when_free.sh` is SFT-specific (it hardcodes sft flags); write a GRPO variant if needed.
- `bhag-delta-gpu` balance on 2026-09-29: **89 h**.

### The 1-hour cap vs 200 GRPO steps
Rough estimate (to be replaced by smoke-test numbers): per GRPO step ~15-25 s rollouts (256 samples, up to 1024
tokens) + ~20-25 s training (128 microbatches of 2) = **~35-50 s/step** early, less once responses shorten.
200 steps ≈ 2-2.5 h, so it does **not** fit in one interactive job. Within 1 h: ~60-90 steps plus evals.
Options: (A) shorter runs (~60-90 steps) compared at equal steps; (B) add checkpoint/resume and chain jobs
(code change: save policy + AdamW state ~18 GB + data position); (C) standard/preempt queue (days / ~12 h waits).

## Handout experiments (§7–8) and what we do with them

| handout problem | what it tests | handout cost | decision |
|---|---|---|---|
| grpo_train_loop | loop works, val reward rises, rollouts over time | — | _pending_ |
| grpo_learning_rate | lr sweep, >= 25% val | 6 H100 h | _pending_ |
| grpo_baselines | no_baseline vs reinforce_with_baseline | 2 H100 h | _pending_ |
| think_about_length_normalization | masked_mean vs masked_normalize (written answer) | — | _pending_ |
| grpo_length_normalization | same, empirically | 2 H100 h | _pending_ (needs a CLI flag) |
| grpo_group_standard_deviation | std normalization on/off (Dr. GRPO) | 2 H100 h | _pending_ |
| grpo_off_policy(_sweep) | several gradient steps per rollout batch + GRPO-Clip | 12 H100 h | _pending_ |
| grpo_off_policy_clip_ablation | GRPO-Clip vs no clip, off-policy | 2 H100 h | _pending_ (needs a new loss type) |
| grpo_prompt_ablation | r1_zero vs question_only prompt | 2 H100 h | _pending_ |
| leaderboard | best MATH score in 4 h | 16 H100 h | skip |

## Results

| run | steps | final val acc | best val acc | final format | wall time | notes |
|---|---|---|---|---|---|---|

## Takeaways

_(filled in as runs finish)_

## TODO
Legend: [x] done, [~] skipped by choice, [-] not planned.
- [ ] Agree on the plan
- [ ] Smoke test (3 GRPO steps, 64 eval examples): check it runs, memory, measured s/step
