# SFT experiments on GSM8K (Qwen2.5-Math-1.5B)

Personal interview-prep exploration based on CS336 A5 §4.3 (`sft_experiment` part 1). Not a course submission.
Rule: run one solid experiment first, look at it, then decide what to run next. Don't mass-queue.

## Setup

- Script: `cs336_alignment/sft/sft.py` (typer CLI, `--help`). Training logic is left unchanged.
- Data: GSM8K instead of MATH. Train `data/gsm8k/train.jsonl` (7473), val `data/gsm8k/test.jsonl` (1319).
  SFT targets are GSM8K solutions reformatted as `... </think> <answer> N </answer>` (calculator `<<..>>` stripped).
- Model: `Qwen/Qwen2.5-Math-1.5B`, cached at `/projects/bhag/gvaccarino/hf_cache` (compute nodes have no internet).
- Layout: policy trains on `cuda:0`, vLLM evaluates on `cuda:1` (one node, 2x A100).
- Eval: r1_zero prompt, **temperature 1.0**, max 1024 tokens, `r1_zero_reward_fn`. At T=1 the accuracy is noisy
  (roughly ±1 pt on 1319 examples), so treat differences of 1–2 pts as within noise.
- Zero-shot baseline (outputs/eval/zero_shot_gsm8k.metrics.json): **accuracy 2.5%, format rate 16.8%**.

### Environment (NCSA Delta)
- venv: `.venv` -> `/work/nvme/bhag/gvaccarino/a5/venv` (home quota is nearly full). uv cache at
  `/work/nvme/bhag/gvaccarino/a5/uv_cache`. Set `UV_CACHE_DIR` to that if you re-sync.
  Do NOT put the venv on `/work/hdd`: on 2026-09-26 `import torch` from there didn't finish in 5+ min, and
  the first smoke job hung in imports for 16 min until I cancelled it.
- flash-attn 2.7.4.post1 installed OK (prebuilt wheel, ~7 min).
- The shell's default `HF_HOME` (`/work/hdd/bghp/huggingface_cache`) is not readable; the job script overrides it.
- SLURM: `jobs/sft.slurm` (account `bhag-delta-gpu`, partition `gpuA100x4-interactive`, 2 GPUs, 16 CPU, 64G,
  **1h max**). Logs in `outputs/slurm/`. Interactive is used on purpose: starts in minutes vs days.
  ```
  sbatch -J <run> jobs/sft.slurm --run-name <run> --attn-implementation flash_attention_2 --n-eval-examples 1319 --seed 0 [...]
  ```
  Queue notes (2026-09-26, fair-share 0.038): `gpuA100x4` estimated start ~10 days out, `gpuA100x4-preempt` ~12 h,
  `gpuA100x4-interactive` (max 1 h) ~5 min. The runs aren't resumable, so preempt without `--requeue`
  risks losing a run. `bhag-delta-gpu` balance: 103 h.
- Time estimate (compute-based, not yet measured): full 3-epoch run ~35-50 min (train ~20-30, 15 evals ~12-15,
  startup ~3-5). The bs64 run with eval-every 25 (~30 evals) may exceed 1h -> use eval-every 50 if needed.
  Whole plan ~4-5 h wall, ~10 GPU-h.
- If CUDA OOM on 40GB A100: add `--gradient-checkpointing`.

## Standard flags
`--attn-implementation flash_attention_2 --n-eval-examples 1319 --seed 0`

## Plan

Phase 1: lr / batch size on the full set (target final acc >= 15%)
| run | flags |
|---|---|
| full_lr1e-5 | `--lr 1e-5 --batch-size 32 --n-epochs 3 --eval-every 50` |
| full_lr2e-5 | `--lr 2e-5 --batch-size 32 --n-epochs 3 --eval-every 50` |
| full_lr5e-5 | `--lr 5e-5 --batch-size 32 --n-epochs 3 --eval-every 50` |
| full_lr2e-5_bs64 | `--lr 2e-5 --batch-size 64 --n-epochs 3 --eval-every 25` |

BEST = highest final eval accuracy.

Phase 2: dataset size with BEST lr/bs (double n-epochs if BEST uses bs 64)
| run | flags |
|---|---|
| n128 | `--n-examples 128 --n-epochs 20 --eval-every 10` |
| n256 | `--n-examples 256 --n-epochs 10 --eval-every 10` |
| n512 | `--n-examples 512 --n-epochs 5 --eval-every 10` |
| n1024 | `--n-examples 1024 --n-epochs 3 --eval-every 10` |
| full | reuse best Phase 1 run |

Plots: `HF_HOME=/projects/bhag/gvaccarino/hf_cache uv run --no-sync python plot_sft_sweep.py --full-run <BEST>` writes
`outputs/sft/{phase1_lr_sweep,dataset_size_sweep}{,_metrics}.png`.

Git: `.gitignore` tracks only `outputs/**/*.md`, `outputs/sft/*.png` and `outputs/sft/<run>/plots/*.png` (no jsonl/json, no smoke runs).

## Results

| run | final acc | best acc | final format | wall time | notes |
|---|---|---|---|---|---|
| **full_lr2e-5** (job 22453531) | **62.7%** | 62.7% | 99.5% | 31.6 min | 701 steps; train 20.8 min, 16 evals 7.6 min |
| smoke (job 22453391) | 26.6% (64 ex) | 26.6% | 62.5% | 3.0 min | 4 steps; step 1.8 s, 64-ex eval 8-12 s, startup ~2.3 min |
| smoke (job 22453139) | FAILED | — | — | 4.5 min | crashed after step-0 eval (acc 3.1%, fmt 25% on 64 ex) |

**Known issue (2026-09-26):** `--attn-implementation flash_attention_2` + default `--param-dtype float32` crashes in
`log_generations` (sft.py:123): the entropy forward runs the fp32 policy *outside* the bf16 autocast, and FlashAttention
rejects fp32 (`RuntimeError: FlashAttention only support fp16 and bf16 data type`). Training steps themselves are under
autocast. Workarounds: `--attn-implementation sdpa` (no code change), or wrap the `log_generations` call in `autocast`.
**Fixed** (user's choice): `log_generations` call in `run_eval` now runs under the same `autocast` as training. Eval path only.

Observation: pre-clip grad norm is 135-295 (loss = per-sequence *sum* of token NLLs, normalize_constant=1), so the
max_grad_norm=1.0 clip is active on every step -> updates are effectively lr * unit-norm gradient.

### full_lr2e-5 notes (2026-09-26)
- Eval curve: 3.1% (step 0) -> 47% by step 50, flat ~47% through epoch 1, jumps to ~56% right after the
  epoch-2 boundary (~step 233), ~60% in epoch 3, final 62.7%. Accuracy steps up at each epoch boundary.
- Format rate 17% -> 96% after 50 steps -> 99.5%. So the first 50 steps mostly teach the *format*; the later gains are
  correctness. Final: 827 correct, 486 formatted-but-wrong, 6 badly formatted.
- Response length 308 -> ~115 tokens; mean token entropy 0.98 -> 0.35 (the model gets more confident).
- Train loss/token 0.76 -> 0.16 while eval is still rising, so no sign of harmful overfitting at 3 epochs.
- Noise check: step 700 vs 701 are nearly identical weights (lr ~0) yet 60.0% vs 62.7% -> T=1 sampling noise
  is ~±1.5-3 pts. Compare runs with that in mind (or add a greedy eval).
- Train loss per token is a staircase: flat within an epoch, sharp drop at each epoch boundary (~0.48 -> 0.31 at
  step ~233, -> 0.18 at ~467). This is the memorization signature: the drops happen exactly when the model starts
  seeing examples it already trained on. Eval accuracy jumps at the same points, so here it still generalizes.
- Response length over the full eval set: correct answers stay ~90 tokens; incorrect answers shrink 230 -> ~130
  but remain longer. Long answers are a signal of the model being wrong.
- Plots: `outputs/sft/phase1_lr_sweep_metrics.png` (acc, format, length correct/incorrect, entropy, loss, grad norm).

## TODO / possible experiments
- [x] Smoke test (128 ex, 1 epoch, 64 eval)
- [x] First solid run: `full_lr2e-5` -> 62.7% final
- [ ] Rest of Phase 1: lr 1e-5, 5e-5, bs64
- [ ] Phase 2 dataset-size sweep
- [ ] Greedy (T=0) eval of the final checkpoint, to separate "learned the task" from sampling noise
- [ ] Filtered SFT / expert iteration (A5 §4.4 / §5): keep only correct model-generated traces
- [ ] Look at failure modes in `outputs/sft/<run>/eval/step_*.jsonl`: format errors vs wrong answers, length
- [x] Entropy / length / loss / grad-norm plots (`*_metrics.png` from plot_sft_sweep.py)
- [ ] Pure bf16 (`--param-dtype bfloat16`) vs fp32 master weights: speed, memory, accuracy
- [ ] Microbatch size / gradient checkpointing throughput tradeoff
- [ ] GRPO on top of the SFT checkpoint (A5 §7)
