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
| grpo_group_standard_deviation | std normalization on/off (Dr. GRPO) | 2 H100 h | **running**: `grpo_no_std` (job 22576274) |
| grpo_off_policy(_sweep) | several gradient steps per rollout batch + GRPO-Clip | 12 H100 h | _pending_ |
| grpo_off_policy_clip_ablation | GRPO-Clip vs no clip, off-policy | 2 H100 h | _pending_ (needs a new loss type) |
| grpo_prompt_ablation | r1_zero vs question_only prompt | 2 H100 h | _pending_ |
| leaderboard | best MATH score in 4 h | 16 H100 h | skip |

## Results

| run | steps | final val acc | best val acc | final format | wall time | notes |
|---|---|---|---|---|---|---|
| **grpo_base** (job 22574790) | 109 (cut by 1 h limit) | **72.7%** (step 100) | 72.7% | 99.9% | 60 min (53 min to last eval) | handout defaults + gradient checkpointing; 959 correct / 359 fmt-wrong / 1 unfmt |
| grpo_base (job 22550804) | 1 | FAILED (OOM) | — | — | 3.8 min | CUDA OOM on policy GPU in step-2 forward; step 0 eval 3.1% / fmt 25.9% |

**OOM on 40 GB A100 (2026-09-29, job 22550804).** Handout defaults assume 80 GB H100s. Step 1 ran
(rollouts 8.8 s + train 18.7 s = 28 s/step), then step 2's forward OOMed: fp32 weights + grads + AdamW state ≈ 24 GB
(Adam state appears after the first optimizer.step, hence step 2), plus microbatch-2 activations with responses up to
~1300 tokens and a 151k-vocab logits tensor (the failed 946 MB alloc). 2.9 GB was reserved-but-unallocated
(fragmentation). Fix, no logic change: `--gradient-checkpointing` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
in `jobs/grpo.slurm`. Rerun = job 22574790. Early signal before the crash: rollout reward 1.6% -> 5.1% at step 2,
zero-std groups 88% -> 66% (most groups of 8 are all-wrong at the start -> no gradient from them).

### Memory breakdown at the OOM (policy GPU, 40 GB A100)
Reconstructed from the config (1.544B params, 28 layers, hidden 1536, MLP 8960, vocab 151,936) and the OOM message
(35.4 GiB allocated + 2.9 GiB reserved-unused, 38.9 GiB used). The failed 946 MiB alloc = one fp32 (2, 816, 151936)
logits-sized tensor -> the crashing microbatch was ~816 tokens.

| component | GiB | share |
|---|---|---|
| fp32 weights (1.544B x 4 B) | 5.75 | 15% |
| fp32 grads, live during accumulation | 5.75 | 15% |
| AdamW m + v (x 8 B), allocated at the first optimizer.step -> why step 1 ran and step 2 OOMed | 11.5 | 30% |
| autocast bf16 weight copies, held until backward | 2.9 | 7% |
| layer activations (~3 MB/token x 1632 tokens) -> what gradient checkpointing removes | ~4.6 | 12% |
| logits + logsumexp/entropy fp32 temporaries (~0.93 GiB each, 4-5 of them) -> NOT helped by checkpointing | ~4.5 | 12% |
| fragmentation | 2.9 | 8% |
| CUDA context | 0.5 | 1% |

With checkpointing + expandable_segments: ~31-33 GiB worst case (T~1200 microbatch). Bigger levers if ever needed:
bf16 params/optimizer (-11.5 GiB), 8-bit Adam (-8.6 GiB), entropy on a subsample / chunked log-softmax.

## Takeaways

### grpo_base (2026-09-30): handout defaults, lr 1e-5, on-policy, reinforce_with_baseline, std norm on
Eval accuracy (1319 test problems, T=1): 3.1% (0) -> 50.3% (10) -> 58.1% (20) -> 64.9% (30) -> 68.1% (40) -> 70.1% (50)
-> 69.6% (60) -> 71.2% (70) -> 72.6% (80) -> 72.2% (90) -> **72.7% (100)**. Plots: `outputs/grpo/base_eval_reward.png`,
`outputs/grpo/base_metrics.png`; example rollouts: `outputs/grpo/grpo_base/rollouts_over_time.md`.

| steps | train reward | zero-std groups | resp. length | token entropy | grad norm (pre-clip) | loss | rollout s | train s |
|---|---|---|---|---|---|---|---|---|
| 1-5 | 0.12 | 0.53 | 219 | 0.68 | 0.35 | -0.07 | 8.9 | 23.7 |
| 6-20 | 0.70 | 0.34 | 125 | 0.15 | 3.0 | -0.05 | 6.0 | 21.1 |
| 21-50 | 0.76 | 0.51 | 129 | 0.16 | 12.4 | -0.31 | 6.2 | 20.9 |
| 51-80 | 0.82 | 0.66 | 126 | 0.08 | 33.5 | -1.95 | 6.5 | 20.8 |
| 81-110 | 0.81 | 0.67 | 118 | 0.05 | 48.7 | -3.69 | 7.6 | 20.7 |

1. **GRPO beats SFT with no reference solutions.** 72.7% after 100 steps vs 62.7% for full-data SFT (3 epochs on
   human solutions) and 64.2% for n128 SFT. It only uses the final number to score its own samples. It passes SFT at
   step ~30 (~16 min). Consistent with the SFT pass@k finding (88% of problems solved by *some* SFT model): the base
   model can already solve most problems, and RL makes it do so reliably.
2. **Two phases.** Steps 0-10: learn the format (eval format 26% -> 93%, train reward 2% -> 68%). Most of the early
   jump is format, as in SFT. Steps 10-100: slow correctness gains (50% -> 73%), flattening after ~50.
3. **Cold start: most groups carry no signal.** At step 1, 88% of groups are all-wrong -> zero advantage, no gradient;
   the model learns from the ~4 questions (of 32) where at least one of 8 samples was right. It escapes by step 3
   (reward 12%, zero-std 47%). Later the opposite happens: by steps 50-110, ~67% of groups are *all-correct* (the
   problems are too easy), so again only ~1/3 of questions produce gradient. GSM8K train is mostly "solved" -> signal
   efficiency drops. (This is what dynamic sampling / filtering to pass-rate 0<p<1 in DAPO addresses.)
4. **Entropy collapse.** Train token entropy 1.0 -> 0.05, far below SFT (full 0.35, n128 0.13). RL sharpens the
   distribution onto answers it already finds; T=1 samples become nearly deterministic. Good for pass@1, but less
   exploration (a known risk: pass@k at large k can shrink).
5. **Loss and grad norm grow, and that's expected, not divergence.** The PG "loss" is not a quantity to minimize to
   0; it went 0 -> -3.7 and pre-clip grad norm 0.35 -> ~49. Cause: with std normalization, a group with 7/8 correct
   gives the single wrong sample advantage -(0.875/0.354) ≈ -2.5, and that wrong sample now has very low probability
   (large negative log-prob). These rare, heavily weighted, low-probability samples dominate loss and gradient. Clipping
   at 1.0 caps the actual update (lr x unit-norm direction), so training stays stable. -> Motivates the
   std-normalization ablation (Dr. GRPO argues this reweighting is a bias).
6. **No length growth.** Responses shrink 219 -> ~120 tokens and stay there (correct ~117, incorrect ~147 on eval).
   No R1-Zero-style "longer reasoning" emerges: GSM8K is easy and 100 steps is short.
7. **Training, not rollouts, dominates step time here.** ~21 s train vs ~6-8 s rollout per step (75% training),
   because of gradient checkpointing, microbatch 2 x 128 accumulation, and short responses. The handout's premise
   for off-policy ("rollouts dominate") is weaker in our setup; relevant if we try off-policy.
8. **Train reward (~81%) > eval (~72%)**: 32-question batches are noisy, and train-set questions vs test set.
9. **Rollouts over time** (rollouts_over_time.md): step 0 answers put prose in the `<answer>` tag or write
   `Answer:` without tags (format 0); by step 50 they're short arithmetic chains with a bare number. Step 100 shows a
   confident mistake (entropy 0.04, $40,000 instead of $70,000): low entropy does not mean correct.


## TODO
Legend: [x] done, [~] skipped by choice, [-] not planned.
- [x] Agree on the plan: base run first (user, 2026-09-29), no checkpoint/resume; ablations decided after
- [~] Separate smoke test -- skipped by choice, went straight to the base run
- [x] grpo_base with gradient checkpointing (job 22574790) -> 72.7% at step 100
- [ ] grpo_no_std: same as grpo_base + `--no-use-std-normalization` (job 22576274). Hypothesis from grpo_base
  takeaway 5: advantages become r - mean(r) in [-1, 1] (no 1/std blow-up for 7/8 or 1/8 groups), so the pre-clip
  grad norm should grow much less; effect on accuracy unclear (Dr. GRPO reports similar or better).
