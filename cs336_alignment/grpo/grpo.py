"""
GRPO on GSM8K (assignment section 7, Algorithm 3). DRAFT: the training loop is left to fill in.

    uv run --no-sync python -m cs336_alignment.grpo.grpo --run-name grpo_base

Layout (same as SFT): policy (HF, trained) on `policy_device`, vLLM on `vllm_device`.
vLLM generates both the training rollouts and the eval samples; policy weights are copied into it
at the start of every GRPO step (so rollouts come from pi_theta_old = current policy).
"""
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.eval.eval import build_prompts, evaluate_vllm, load_gsm8k, load_prompt_template
from cs336_alignment.grpo.group_normalization import compute_group_normalized_rewards
from cs336_alignment.grpo.grpo_microbatch import grpo_microbatch_train_step
from cs336_alignment.grpo.masked_mean import masked_mean
from cs336_alignment.sft.log_generations import log_generations
from cs336_alignment.sft.response_log_probs import get_response_log_probs
from cs336_alignment.sft.tokenize_prompt_output import tokenize_prompt_and_output
from cs336_alignment.utils.local_logger import LocalLogger
from cs336_alignment.utils.vllm_utils import init_vllm, load_policy_into_vllm_instance

# prompt name -> (reward fn, vLLM stop strings). question_only has no tags, so no </answer> stop.
PROMPT_SETUPS = {
    "r1_zero": (r1_zero_reward_fn, ["</answer>"]),
    "question_only": (question_only_reward_fn, None),
}


# ---------------------------------------------------------------------------------------------
# Helpers (plumbing)
# ---------------------------------------------------------------------------------------------

def iterate_question_batches(
    pairs: list[tuple[str, str]], n_prompts: int, seed: int
) -> Iterator[list[tuple[str, str]]]:
    """Endless iterator of `n_prompts` (prompt, ground_truth) pairs, reshuffled each pass over the data."""
    rng = random.Random(seed)
    while True:
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for start in range(0, len(order) - n_prompts + 1, n_prompts):
            yield [pairs[i] for i in order[start:start + n_prompts]]


def sample_rollouts(
    llm: LLM,
    question_batch: list[tuple[str, str]],
    sampling_params: SamplingParams,
) -> tuple[list[str], list[str], list[str]]:
    """
    Generate `sampling_params.n` (= group_size) responses per prompt.

    Returns three flat lists of length n_prompts * group_size, ordered group by group
    (prompt 0's G responses, then prompt 1's, ...), i.e. the "(B G)" layout that
    compute_group_normalized_rewards expects:
        repeated_prompts, responses, repeated_ground_truths
    """
    prompts = [p for p, _ in question_batch]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    repeated_prompts, responses, repeated_gts = [], [], []
    for (prompt, gt), out in zip(question_batch, outputs):
        for completion in out.outputs:
            repeated_prompts.append(prompt)
            responses.append(completion.text)
            repeated_gts.append(gt)
    return repeated_prompts, responses, repeated_gts


def main(
    # --- run / io ---
    run_name: str = "grpo",
    output_dir: str = "outputs/grpo",
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",  # or a saved SFT checkpoint dir
    train_path: str = "data/gsm8k/train.jsonl",
    val_path: str = "data/gsm8k/test.jsonl",
    prompt_name: str = "r1_zero",               # "r1_zero" | "question_only" (prompt ablation, train AND eval)
    seed: int = 0,
    # --- GRPO (handout defaults, section 7.2) ---
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,              # responses per GRPO step = n_prompts * group_size
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,               # no empty responses (NaN in masked_mean)
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,          # 1 + train_batch_size == rollout_batch_size -> on-policy
    train_batch_size: int = 256,                # responses per optimizer step
    gradient_accumulation_steps: int = 128,     # microbatch = train_batch_size // this = 2
    loss_type: str = "reinforce_with_baseline", # "no_baseline" | "reinforce_with_baseline" | "grpo_clip"
    use_std_normalization: bool = True,
    cliprange: float = 0.2,                     # only used by grpo_clip
    max_grad_norm: float = 1.0,
    # --- model / memory ---
    param_dtype: str = "float32",               # fp32 master weights + bf16 autocast (see SFT notes)
    attn_implementation: str = "flash_attention_2",
    gradient_checkpointing: bool = False,
    policy_device: str = "cuda:0",
    vllm_device: str = "cuda:1",
    gpu_memory_utilization: float = 0.85,
    # --- eval ---
    eval_every: int = 10,                       # GRPO steps; also at step 0 and at the end
    n_eval_examples: int | None = 1024,         # handout: >= 1024 to compare hyperparameters
    n_log_generations: int = 16,
    save_model: bool = False,
):
    config = dict(locals())
    run_dir = Path(output_dir) / run_name
    logger = LocalLogger(run_dir, config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------------- sanity checks + derived sizes (from the handout) ----------------
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    assert rollout_batch_size % train_batch_size == 0, "rollout_batch_size must be divisible by train_batch_size"
    n_train_steps_per_rollout_batch = rollout_batch_size // train_batch_size   # optimizer steps per epoch
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    assert loss_type in ("no_baseline", "reinforce_with_baseline", "grpo_clip"), loss_type
    if loss_type == "grpo_clip":
        # Not an error, but with 1 optimizer step per rollout batch the ratio is always 1 (= REINFORCE).
        if epochs_per_rollout_batch == 1 and n_train_steps_per_rollout_batch == 1:
            print("WARNING: grpo_clip in the on-policy setting reduces to reinforce_with_baseline")
    print(
        f"{n_prompts_per_rollout_batch} prompts x {group_size} = {rollout_batch_size} rollouts/step, "
        f"{n_train_steps_per_rollout_batch} optimizer steps/epoch x {epochs_per_rollout_batch} epochs, "
        f"microbatch {micro_train_batch_size}"
    )

    # ---------------- data ----------------
    reward_fn, stop = PROMPT_SETUPS[prompt_name]
    template = load_prompt_template(prompt_name)
    train_pairs = build_prompts(load_gsm8k(train_path), template)          # [(prompt, ground_truth)]
    val_pairs = build_prompts(load_gsm8k(val_path, limit=n_eval_examples), template)
    question_batches = iterate_question_batches(train_pairs, n_prompts_per_rollout_batch, seed)

    # ---------------- models ----------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, param_dtype),
        attn_implementation=attn_implementation,
    ).to(policy_device)
    if gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.train()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if param_dtype == "float32" else nullcontext()
    )

    llm = init_vllm(model_name, vllm_device, seed, gpu_memory_utilization)
    rollout_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        stop=stop,
        include_stop_str_in_output=True,
    )
    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=stop, include_stop_str_in_output=True,
        seed=seed,  # same random draws at every eval -> less noise between eval points
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95)
    )

    # ---------------- eval helper ----------------
    def run_eval(step: int):
        """Copies the current policy into vLLM, evaluates on val_pairs, logs + plots eval/*."""
        t0 = time.time()
        load_policy_into_vllm_instance(policy, llm)
        metrics = evaluate_vllm(
            llm, eval_sampling_params, val_pairs, reward_fn,
            output_path=str(run_dir / "eval" / f"step_{step:05d}.jsonl"), batch_size=None,
        )
        with autocast:  # flash_attention_2 needs bf16 activations for the entropy forward
            gen = log_generations(
                llm, policy, tokenizer,
                [p for p, _ in val_pairs[:n_log_generations]],
                [g for _, g in val_pairs[:n_log_generations]],
                reward_fn, eval_sampling_params,
                output_path=str(run_dir / "generations.jsonl"), step=step,
            )
        logger.log("eval", step, {
            "answer_reward": metrics["accuracy"],
            "format_reward": metrics["format_rate"],
            "mean_token_entropy": gen["summary"]["mean_token_entropy"],
            "response_length": gen["summary"]["response_length"],
            "response_length_correct": gen["summary"]["response_length_correct"],
            "response_length_incorrect": gen["summary"]["response_length_incorrect"],
            "eval_time_s": time.time() - t0,
            "wall_clock_s": time.time() - t_start,
        })
        logger.plot("eval")
        print(f"[eval step {step}] answer_reward={metrics['accuracy']:.3f} format={metrics['format_rate']:.3f}")

    # ---------------- train loop ----------------
    t_start = time.time()
    run_eval(0)
    train_step = 0  # optimizer-step counter (x-axis of train/*); grpo_step is the x-axis of rollout/*

    def get_microbatch(batch: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Rows `idx` of the tokenized rollout batch, moved to policy_device and trimmed to their own
        longest sequence (the batch is padded to the longest of all 256 rollouts; with right padding,
        everything after the last response token of these rows is pure padding).
        """
        rows = {k: v[idx] for k, v in batch.items()}
        mask = rows["response_mask"]
        positions = torch.arange(mask.shape[1])
        length = int((mask * (positions + 1)).max())  # last response position + 1 over these rows
        return {
            k: (v[:, :length] if v.dim() == 2 and v.shape[1] == mask.shape[1] else v).to(policy_device)
            for k, v in rows.items()
        }

    for grpo_step in range(1, n_grpo_steps + 1):
        t0 = time.time()

        # 1. Rollouts from pi_theta_old (= current policy)
        load_policy_into_vllm_instance(policy, llm)
        question_batch = next(question_batches)
        repeated_prompts, responses, repeated_gts = sample_rollouts(llm, question_batch, rollout_sampling_params)
        rollout_time = time.time() - t0

        # 2. Rewards + advantages, (rollout_batch_size,) -> (rollout_batch_size, 1) for the losses
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn, responses, repeated_gts, group_size, advantage_eps, use_std_normalization
        )

        # 3. Tokenize all rollouts once; microbatches are sliced out (and trimmed) below
        obj = tokenize_prompt_and_output(repeated_prompts, responses, tokenizer)
        rollout_batch = {
            "input_ids": obj["input_ids"],
            "labels": obj["labels"],
            "response_mask": obj["response_mask"],
            "advantages": advantages.unsqueeze(-1).float(),
            "raw_rewards": raw_rewards.unsqueeze(-1).float(),
        }
        response_lengths = obj["response_mask"].sum(dim=-1).float()
        correct = raw_rewards == 1.0

        logger.log("rollout", grpo_step, {
            "reward": reward_meta["reward_mean"],
            "format_reward": reward_meta["format_reward_mean"],
            "frac_zero_std_groups": reward_meta["frac_zero_std_groups"],
            "response_length": response_lengths.mean().item(),
            "response_length_correct": response_lengths[correct].mean().item() if correct.any() else None,
            "response_length_incorrect": response_lengths[~correct].mean().item() if (~correct).any() else None,
            "rollout_time_s": rollout_time,
        })

        # 4. Old log-probs (grpo_clip only): once per rollout batch, before any update, no gradient
        if loss_type == "grpo_clip":
            old_log_probs = torch.zeros(obj["labels"].shape, dtype=torch.float32)
            with torch.no_grad(), autocast:
                for start in range(0, rollout_batch_size, micro_train_batch_size):
                    idx = torch.arange(start, start + micro_train_batch_size)
                    mb = get_microbatch(rollout_batch, idx)
                    lp = get_response_log_probs(policy, mb["input_ids"], mb["labels"])["log_probs"]
                    old_log_probs[idx, : lp.shape[1]] = lp.float().cpu()
            rollout_batch["old_log_probs"] = old_log_probs

        # 5. Epochs x train batches x microbatches
        for epoch in range(epochs_per_rollout_batch):
            perm = torch.randperm(rollout_batch_size)  # mixes groups across train batches when off-policy
            for b in range(n_train_steps_per_rollout_batch):
                t_step = time.time()
                batch_idx = perm[b * train_batch_size:(b + 1) * train_batch_size]
                loss_sum, entropy_sum, clip_sum = 0.0, 0.0, 0.0

                for m in range(gradient_accumulation_steps):
                    idx = batch_idx[m * micro_train_batch_size:(m + 1) * micro_train_batch_size]
                    mb = get_microbatch(rollout_batch, idx)
                    with autocast:
                        out = get_response_log_probs(
                            policy, mb["input_ids"], mb["labels"], return_token_entropy=True
                        )
                    # Entropy is only logged: detach right away so its (B, T, V) graph is freed.
                    entropy_sum += masked_mean(out.pop("token_entropy").detach().float(), mb["response_mask"]).item()
                    loss, meta = grpo_microbatch_train_step(
                        out["log_probs"].float(),
                        mb["response_mask"],
                        gradient_accumulation_steps,
                        loss_type,
                        raw_rewards=mb["raw_rewards"],
                        advantages=mb["advantages"],
                        old_log_probs=mb.get("old_log_probs"),
                        cliprange=cliprange,
                    )
                    loss_sum += loss.item()  # already / gradient_accumulation_steps -> sums to the batch mean
                    if "clipped" in meta:
                        clip_sum += masked_mean(meta["clipped"].float(), mb["response_mask"]).item()

                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_step += 1

                logger.log("train", train_step, {
                    "loss": loss_sum,
                    "grad_norm": grad_norm.item(),  # before clipping
                    "token_entropy": entropy_sum / gradient_accumulation_steps,
                    "clip_fraction": clip_sum / gradient_accumulation_steps if loss_type == "grpo_clip" else None,
                    "grpo_step": grpo_step,
                    "epoch": epoch,
                    "step_time_s": time.time() - t_step,
                    "wall_clock_s": time.time() - t_start,
                })

        print(
            f"[grpo step {grpo_step}/{n_grpo_steps}] reward={reward_meta['reward_mean']:.3f} "
            f"zero_std_groups={reward_meta['frac_zero_std_groups']:.2f} "
            f"len={response_lengths.mean():.0f} time={time.time() - t0:.0f}s"
        )
        if grpo_step % 5 == 0:
            logger.plot("train")
            logger.plot("rollout")

        # 6. Eval
        if grpo_step % eval_every == 0 or grpo_step == n_grpo_steps:
            run_eval(grpo_step)

    logger.plot()
    if save_model:
        policy.save_pretrained(run_dir / "model")
        tokenizer.save_pretrained(run_dir / "model")


if __name__ == "__main__":
    typer.run(main)
