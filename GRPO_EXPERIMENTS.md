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
  QOS: 2 submitted / 1 running per user; **changed to 1 submitted / 1 running by 2026-10-01** -> use
  `jobs/submit_grpo_when_free.sh <run> [flags]` detached (setsid nohup) to queue the next run; log `outputs/slurm/submitter.log`). Logs in `outputs/slurm/`, runs in `outputs/grpo/<run>/`.
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
| grpo_baselines | no_baseline vs reinforce_with_baseline | 2 H100 h | ran `grpo_no_baseline`: 45% vs 72.7% with baseline |
| think_about_length_normalization | masked_mean vs masked_normalize (written answer) | — | _pending_ |
| grpo_length_normalization | same, empirically | 2 H100 h | _pending_ (needs a CLI flag) |
| grpo_group_standard_deviation | std normalization on/off (Dr. GRPO) | 2 H100 h | ran 2 seeds: seed 0 **collapsed**, seed 1 stable at 75.3% (both show a drift episode at steps ~21-35) |
| grpo_off_policy(_sweep) | several gradient steps per rollout batch + GRPO-Clip | 12 H100 h | _pending_ |
| grpo_off_policy_clip_ablation | GRPO-Clip vs no clip, off-policy | 2 H100 h | _pending_ (needs a new loss type) |
| grpo_prompt_ablation | r1_zero vs question_only prompt | 2 H100 h | ran `grpo_question_only` (x2): 58% zero-shot -> 84% |
| leaderboard | best MATH score in 4 h | 16 H100 h | skip |

## Results

| run | steps | final val acc | best val acc | final format | wall time | notes |
|---|---|---|---|---|---|---|
| **grpo_question_only** (job 22601289) | 74 (cut by 1 h limit) | **84.2%** (step 70) | 84.3% @60 | 98.9% | 60 min | `--prompt-name question_only`; zero-shot (step 0) already 58.0% |
| grpo_question_only_rerun (job 22607113) | 78 (cut by 1 h limit) | 83.9% (step 70) | 83.9% | 99.1% | 60 min | accidental same-seed duplicate (my error); useful as a nondeterminism check |
| grpo_no_baseline (job 22597253) | 119 (cut by 1 h limit) | 45.4% (step 110) | 52.1% @100 | 98.8% | 60 min | `--loss-type no_baseline`; plateaus ~45%, answers shrink to ~40 tokens; 599 correct / 704 fmt-wrong / 16 unfmt |
| **grpo_no_std_seed1** (job 22584780) | 95 (cut by 1 h limit) | **75.3%** (step 90) | 75.3% | 99.5% | 60 min | no std norm, seed 1; stable, but a near-miss at steps ~21-35 (see notes); first run with rollouts.jsonl |
| grpo_no_std (job 22576274) | 61 (OOM) | 4.2% (step 60) | **61.1% @30** | 6.4% | 52 min | `--no-use-std-normalization`; collapsed after step ~31 (format drift -> 1024-token garbage), then OOM |
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

### grpo_no_std (2026-09-30): std normalization off -> training collapse (single seed)
Eval: 3.1% (0) -> 45.0 (10) -> 56.3 (20) -> **61.1 (30)** -> 45.5 (40) -> **3.6 (50)** -> 4.2 (60); format 93% (10)
-> 97.6 (20) -> 93.4 (30) -> 65.4 (40) -> 4.9 (50). Plots: `outputs/grpo/std_ablation_{eval_reward,metrics}.png`.

- **Steps 0-30: same as base, slightly slower** (61.1% vs 64.9% at step 30). Pre-clip grad norm stayed lower
  (~1-3 vs ~5-9 for base), as hypothesized: advantages in [-1, 1] without the 1/std blow-up.
- **Steps ~31-50: two-stage collapse.** vLLM stops at `</answer>` OR at EOS (`<|endoftext|>`), so a missing tag
  does NOT by itself force a 1024-token runaway (corrected 2026-09-30; my first write-up claimed it did).
  Eval answers not ending in `</answer>` -> (ran to ~1024 / stopped via EOS):
  step 30: 72 (4 / 68) · step 40: 366 (59 / 307, median ~180 tok) · step 50: 982 (502 / 480, median ~570 tok) ·
  step 60: 1069 (587 / 482). (Base model at step 0 for reference: 523 (45 / 478).)
  - *Stage 1 (steps ~31-40), format slip:* correct reasoning, ends `</think> $18` + EOS, short. Reward 0 -> eval
    61% -> 45% with no runaway yet.
  - *Stage 2 (steps ~40-50), losing the ability to stop:* as entropy climbs, EOS stops being emitted reliably ->
    ~half the untagged answers hit max_tokens=1024 and the rest get long too (loops like `</think> First, let's...`,
    then near-random text). Train token entropy 0.2 -> **6.6** (uniform over vocab = 11.9), rollout length
    150 -> ~700, reward -> 2%, all-wrong groups -> ~90% (no signal left to recover), seconds/step 30 -> 58.
- **Crash:** OOM at step ~61 on a 1.93 GiB fp32 logits alloc (microbatch of 2 x ~1700 re-tokenized garbage tokens);
  a symptom of the collapse, not a separate bug. Fragmentation was no longer an issue (113 MiB reserved-unused).
- **Why it self-reinforces (mechanism, fairly confident):** long zero-reward garbage gets a negative advantage; with
  masked_mean each token's weight is A/len, so a 1024-token response is penalized only weakly per token, and pushing
  down sampled tokens spreads probability mass to other tokens -> entropy goes *up* (the opposite of the collapse in
  grpo_base). More entropy -> more garbage (and fewer EOS) -> more long negative samples. This explains stage 2;
  it only kicks in once responses are already failing and getting long. Stage 1's trigger is unexplained.
- **Trigger analysis (2026-09-30).** Ruled out: (a) the stop string (vLLM also stops at EOS); (b) a length bias on
  shared tokens (with masked_mean, weight = A/len; if wrong answers were *shorter* the shared `<answer>` token would
  get a net push down, but wrong answers were *longer* in both runs, e.g. 191 vs 136 tokens at step 31). Untagged eval
  answers: base 3.6% (10) -> 0.7 (20) -> **0.1 (30)**; no_std 6.0 -> 1.4 -> **3.9 (30)** -> 27.6 (40). So in no_std the
  pressure against slips stalled after step 20. Best-supported explanation: rare slips sit in nearly-solved groups
  (7/8 correct); with std norm each non-degenerate group gets ~equal weight (total |A| ∝ sqrt(p(1-p))), without it
  weight ∝ p(1-p), so a 7/8 group counts ~1.5x less relative to a 50/50 group -> the "fix the rare slip on an easy
  question" signal is down-weighted (the flip side of Dr. GRPO's difficulty-bias fix). This explains *not
  suppressing* slips, not their *growth* 4% -> 28% in 10 steps; concurrently no_std's correct answers were getting
  longer (116 -> 162 tok by step 28 vs base ~105-114) and entropy started rising ~step 25-34, but order is unclear.
- **Is it caused by removing std normalization?** Not established. One seed per
  config; identical rollouts at step 1, then chaotic divergence. It could be std-norm-specific or seed luck.
  To attribute it: rerun `grpo_no_std` (and/or `grpo_base`) with `--seed 1`.
- **Early warning:** train-side entropy (0.27 -> 2.5) and rollout length (150 -> 250) jumped at step ~33-34, while
  the step-30 eval still looked fine (61%). Entropy and length are the metrics to alert on; eval every 10 steps lagged.

### grpo_no_std_seed1 (2026-10-01): no std norm, seed 1 -> stable, 75.3%, but the same drift episode
Eval: 2.9% (0) -> 43.9 (10) -> 55.8 (20) -> 63.4 (30) -> 69.5 (40) -> 72.3 (50) -> 72.6 (60) -> 74.6 (70) -> 75.0 (80)
-> **75.3 (90)**. Plots (3 runs overlaid): `outputs/grpo/std_ablation_{eval_reward,metrics}.png`.

| steps | run | train reward | zero-std groups | length (correct / wrong) | token entropy | grad norm |
|---|---|---|---|---|---|---|
| 21-35 | base | 0.73 | 0.47 | 114 / 153 | 0.14 | 6.3 |
| 21-35 | no_std seed 0 | 0.70 | 0.43 | 145 / 232 | 0.69 | 4.4 |
| 21-35 | no_std seed 1 | **0.54** | 0.18 | 140 / 311 | **0.77** | 7.2 |
| 51-80 | base | 0.82 | 0.66 | 115 / 175 | 0.08 | 33.5 |
| 51-80 | no_std seed 1 | 0.68 | 0.35 | 138 / 227 | 0.41 | 22.7 |
| 81-95 | base | 0.82 | 0.64 | 109 / 151 | 0.05 | 54.0 |
| 81-95 | no_std seed 1 | 0.78 | 0.58 | 153 / 299 | 0.13 | 12.1 |

- **The collapse was neither pure luck nor inevitable.** Seed 1 went through the *same* episode at steps ~21-35 and
  recovered. From `rollouts.jsonl` (training rollouts): untagged answers 4.1% (steps 11-20) -> **13.2% (21-30)** ->
  6.5% (31-40) -> 1.7% (41-60); train reward dipped to ~0.54, entropy spiked to ~1.6 around step 25, wrong answers got
  long (311 tok). Seed 0 had the same drift starting ~step 31 and did not recover. Base (seed 0) never showed it
  (untagged -> 0.1%, entropy ~0.14 throughout). So 2/2 no_std seeds show a fragile phase; base 0/1. Suggestive, not
  proof: there is no base seed-1 run for a matched pair.
- **My "nearly-solved groups are down-weighted" explanation is NOT supported by the rollout log.** Untagged answers
  in steps 21-30 were spread over all group types (correct-per-group k=0..7: 43, 76, 42, 56, 34, 54, 25, 9), not
  concentrated in 7/8 groups, and they did receive negative advantages (mean -0.35 to -0.46). The trigger of the
  drift is still not identified. What is reproducible is the signature: entropy and wrong-answer length rise
  together with untagged answers.
- **Stable-state differences vs base (different seeds, so +-2-3 pts of noise):** accuracy 75.3% vs 72.2% at step 90;
  entropy stays higher for longer (0.41 vs 0.08 at steps 51-80; 0.13 vs 0.05 at the end) -> less entropy collapse;
  fewer zero-std groups (0.35 vs 0.66 at steps 51-80) -> more questions still give gradient; answers longer (correct
  ~140-150 vs ~110-115 tokens); pre-clip grad norm lower (12 vs 54 at the end), as predicted.
- **Net:** removing std normalization here looks like a trade: more exploration / less collapse and equal-or-better
  accuracy, but a less stable phase around steps 20-35 that can be fatal (1 of 2 seeds).
- Rollout logging works: 24,320 rows (95 steps x 256), 33 MB.

### grpo_no_baseline (2026-10-01): raw 0/1 rewards as weights -> ~45%, answers collapse to ~40 tokens
Eval: 3.1% (0) -> 27.7 (10) -> 40.7 (20) -> 48.2 (30) -> 48.1 (40) -> 45.3 (50) -> 42.7 (60) -> 44.5 (70) -> 44.6 (80)
-> 45.3 (90) -> 52.1 (100) -> 45.4 (110). Base at the same steps: 50.3, 58.1, 64.9, 68.1, 70.1, 69.6, 71.2, 72.6, 72.2,
72.7. Plots: `outputs/grpo/baseline_ablation_{eval_reward,metrics}.png`.

| steps | run | train reward | format | length correct / wrong | token entropy | grad norm | loss |
|---|---|---|---|---|---|---|---|
| 6-20 | base | 0.70 | 0.93 | 110 / 152 | 0.15 | 3.0 | -0.05 |
| 6-20 | no_baseline | 0.48 | 0.91 | 102 / 111 | 0.54 | 0.32 | +0.19 |
| 21-50 | base | 0.76 | 0.98 | 117 / 171 | 0.16 | 12.4 | -0.31 |
| 21-50 | no_baseline | 0.65 | 0.98 | 98 / 99 | 0.42 | 0.37 | +0.24 |
| 81-110 | base | 0.81 | 0.95 | 109 / 155 | 0.05 | 48.7 | -3.69 |
| 81-110 | no_baseline | 0.54 | 0.99 | **39 / 47** | 0.21 | 0.79 | +0.10 |

- **~27 points worse than with the baseline** (45% vs 72.7%), and it's not just slower: accuracy peaks ~48% at step
  30-40, then drifts down while train reward falls 0.65 -> 0.54.
- **Format is learned equally well** (99%); what fails is correctness: 704 well-formatted wrong answers at step 110.
- **The model stops reasoning.** Response length 120 -> ~40 tokens. Step-110 samples:
  `16-3-4=9</think> <answer>18</answer>`, ` 2+1=3</think> <answer>3</answer>`. Terse one-line arithmetic, right on easy
  questions and wrong on anything needing more steps.
- **Why (mechanism):** with no baseline the loss weight is the raw reward: +1 for every correct answer, 0 for every
  wrong one. So (a) nothing is ever pushed down: a terse wrong guess costs nothing; (b) the gradient is "imitate your
  own correct samples", dominated by easy questions (all 8 correct -> 8 positive samples, hard questions -> few or
  none); (c) with masked_mean each token's weight is 1/len, so short correct answers get more gradient per token, and
  with only positive weights there is no opposing term -> drift toward ever-shorter answers. With a baseline, an
  all-correct group gets advantage 0 (no update) and wrong answers get negative advantage, which removes (a)-(c).
- **Update size differs too (a confound):** pre-clip grad norm is 0.15-0.8 here (< 1, so *not* clipped -> smaller
  steps) vs 3-50 for base (always clipped to 1). Part of the early gap (27.7% vs 50.3% at step 10) may be step size;
  the later decline is not.
- **Prediction check:** I expected faster entropy collapse; wrong. Entropy stays *higher* (0.21 vs 0.05 at the end),
  consistent with the smaller, positive-only updates. "zero-std groups" is not meaningful for this loss (all-correct
  groups still produce gradient).
- **Interview version:** the baseline is not only variance reduction in theory; with 0/1 rewards it is what turns
  "wrong" into a negative signal and "everyone was right" into no signal. Without it you get reward-weighted
  self-imitation, which here drifts to short, lazy answers.

### grpo_question_only (2026-10-01): bare-question prompt -> 58% zero-shot, 84% after 70 steps
Prompt = just the question; reward = `question_only_reward_fn` (needs a `\\boxed{}` answer); no stop string (ends at
EOS or 1024 tokens); same prompt and reward for train and eval. Plots: `outputs/grpo/prompt_ablation_{eval_reward,metrics}.png`.

| eval step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|
| question_only (job 22601289) | 58.0 | 73.2 | 80.6 | 82.8 | 83.2 | 83.2 | 84.3 | 84.2 |
| question_only rerun (same seed) | 58.0 | 77.4 | 80.6 | 80.5 | 81.0 | 81.9 | 82.0 | 83.9 |
| r1_zero prompt (grpo_base) | 3.1 | 50.3 | 58.1 | 64.9 | 68.1 | 70.1 | 69.6 | 71.2 |

- **The prompt is worth more than all the training we did with r1_zero.** The *untrained* model scores 58% with the
  bare question vs 3% with the r1_zero prompt, and after 70 GRPO steps 84% vs 71%. With r1_zero most of the RL budget
  goes into learning an unfamiliar output format; with the bare question the model is already in its pretraining
  format (step-0 format reward 0.89: it writes `\\boxed{}` on its own).
- **What the model does natively:** long step-by-step text (~350-400 tokens vs ~120 with r1_zero), often followed by
  Python code and a *hallucinated* ```` ```output ```` block (nothing is executed), then `\\boxed{answer}`. This is
  Qwen2.5-Math's tool-integrated-reasoning style from pretraining. Wrong answers often "confirm" a wrong number via
  the fake output.
- **RL barely moves the policy, yet accuracy rises 26 points.** Pre-clip grad norm is 0.06-0.16 (never clipped, tiny
  updates), entropy starts low (0.22) and drifts to 0.09, length stays ~380. Gains come from fixing errors (boxed
  but wrong: 408 -> 201 on the 1319 test problems; no `\\boxed`: 144 -> 11), not from a change of style.
- **Train-test gap suggests GSM8K train was in the model's pretraining data.** Train rollout reward is 0.70 at
  steps 1-5 (before any real learning) vs 0.58 on test at step 0; at the end 0.93 vs 0.84. It cannot be RL
  overfitting: 74 steps x 32 questions = 2,368 < 7,473, so no training question was seen twice. The r1_zero run shows
  the same gap (0.82 train vs 0.72 test). Caveat: 160 train questions at steps 1-5, so +-4 pts.
- **Saturation:** by steps 51-74, ~80% of groups are all-correct (zero std) -> little gradient left on GSM8K train.
- **Same-seed repeat (accidental):** the two runs differ by up to 4 pts at a given eval step (77.4 vs 73.2 at step
  10, 80.5 vs 82.8 at step 30) and 0.3 pts at step 70. The seed does not make runs deterministic (vLLM sampling);
  +-2-3 pts is the run-to-run noise to keep in mind for every comparison in this file.
- **Cost:** steps ~35 s (rollout 11 s + train 24 s) and evals ~75 s (long answers), so only ~74 steps fit in the hour.
  No OOM.
- **Interview version:** RL results are only meaningful relative to the prompt/format the base model was pretrained
  on. A big "RL gain" can be mostly format learning (r1_zero: 3% -> 72%), and the same model with its native prompt
  starts at 58%. Always report the zero-shot number for the prompt you train with.

**Process error (mine):** two auto-submit scripts were alive at once (the first had not died as I assumed), so
`grpo_question_only` was submitted twice and both jobs wrote into the same run folder (~2 GPU-h wasted). Files were
split on 2026-10-02 into `grpo_question_only/` and `grpo_question_only_rerun/` (`*.mixed_backup` kept; the first
run's per-step `eval/step_*.jsonl` were overwritten and are lost). Before starting a submitter, check
`ps -u $USER -o pid,args | grep "[s]ubmit_grpo"`, and never reuse a run name.

### Entropy collapse: why, is it a problem, fixes (discussion, 2026-09-30)
- **Why.** For softmax policies, the entropy change per update is ~ -Cov(log pi(token), advantage) (Cui et al. 2025,
  "The Entropy Mechanism of RL for Reasoning LMs"). Likely tokens are more often correct -> positive covariance ->
  entropy falls (rich-get-richer). Nothing pushes back: 0/1 outcome reward, no KL, no entropy term; negative
  advantages (which spread mass) become rare once reward is ~80%. Reward-irrelevant wording locks into one template
  (steps 50 and 100 rollouts are near word-for-word identical).
- **Is it a problem?** Not for pass@1 so far (accuracy kept rising). But: (1) exploration dies -> 8 identical
  samples = zero-std group = no gradient, and always-wrong questions never produce a correct sample to learn from;
  entropy < 0.1, zero-std groups ~67% and the accuracy plateau all coincide around steps 50-80 (consistent, not
  proven); (2) pass@k at large k can drop below the base model (Yue et al. 2025); (3) confident errors / worse
  calibration; (4) less diversity left for further RL.
- **Fixes:** clip-higher (DAPO; off-policy only), dynamic sampling / harder data (DAPO), entropy bonus (better:
  adaptive to a target entropy), Clip-Cov / KL-Cov (Cui et al.), update only high-entropy forking tokens (Wang et al.
  2025), up-weight negative samples (Zhu et al. 2025), KL to reference (slows learning too), higher rollout
  temperature / larger G. Modern recipes treat entropy as the main health metric.
- **Possible test (not planned yet):** pass@k (k=8/16) of base vs step-100 model; needs --save-model + small eval code.

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
- [x] grpo_no_std: same as grpo_base + `--no-use-std-normalization` (job 22576274) -> peaked 61.1% @30, collapsed to ~4%, OOM at step 61.
- [x] grpo_no_std_seed1 (job 22584780), `--seed 1` + rollout logging -> stable, 75.3% at step 90; near-miss at steps ~21-35.
  takeaway 5: advantages become r - mean(r) in [-1, 1] (no 1/std blow-up for 7/8 or 1/8 groups), so the pre-clip
  grad norm should grow much less; effect on accuracy unclear (Dr. GRPO reports similar or better).
- [x] grpo_no_baseline (job 22597253) -> 45.4% at step 110 (best 52.1%), vs 72.7% for base. See notes.
- [x] grpo_question_only (job 22601289) -> 84.2% at step 70 (zero-shot 58.0%); duplicate rerun 22607113 -> 83.9%. See notes.