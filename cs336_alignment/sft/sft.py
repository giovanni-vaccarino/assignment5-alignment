"""
SFT on GSM8K (assignment section 4.3, Algorithm 1). DRAFT: not run yet.

    uv run --no-sync python -m cs336_alignment.sft.sft --n-examples 512 --run-name sft_512

Layout: policy (HF, trained) on `policy_device`, vLLM (eval only) on `vllm_device`.
Every `eval_every` optimizer steps the policy weights are copied into vLLM and evaluated.
"""
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval.eval import build_prompts, evaluate_vllm, load_gsm8k, load_prompt_template
from cs336_alignment.sft.data import iterate_sft_batches, load_gsm8k_sft
from cs336_alignment.sft.log_generations import log_generations
from cs336_alignment.sft.response_log_probs import get_response_log_probs
from cs336_alignment.sft.sft_microbatch import sft_microbatch_train_step
from cs336_alignment.utils.local_logger import LocalLogger
from cs336_alignment.utils.vllm_utils import init_vllm, load_policy_into_vllm_instance


def main(
    # --- run / io ---
    run_name: str = "sft",
    output_dir: str = "outputs/sft",
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    train_path: str = "data/gsm8k/train.jsonl",
    val_path: str = "data/gsm8k/test.jsonl",
    seed: int = 0,
    # --- data ---
    n_examples: int | None = None,        # None = full train set (7473); sweep {128, 256, 512, 1024, None}
    # --- optimization ---
    n_epochs: float = 3.0,                # total steps = ceil(n_examples * n_epochs / batch_size)
    batch_size: int = 32,                 # examples per optimizer step (effective batch)
    microbatch_size: int = 2,             # examples per forward/backward; grad_accum = batch_size // microbatch_size
    lr: float = 2e-5,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.05,           # linear warmup, then cosine decay to 0
    max_grad_norm: float = 1.0,
    normalize_constant: float = 1.0,      # passed to masked_normalize (sum of token NLLs / this)
    # --- model / memory ---
    param_dtype: str = "float32",         # "float32" = fp32 master weights + bf16 autocast; "bfloat16" = pure bf16
    attn_implementation: str = "sdpa",    # "flash_attention_2" on the rented GPU
    gradient_checkpointing: bool = False,
    policy_device: str = "cuda:0",
    # --- eval ---
    vllm_device: str = "cuda:1",
    vllm_gpu_memory_utilization: float = 0.85,  # lower it (e.g. 0.3) if vllm_device == policy_device
    eval_every: int = 25,                 # optimizer steps; also evaluates at step 0 and at the end
    n_eval_examples: int | None = 500,    # subset for in-loop evals; None = full val set
    n_log_generations: int = 16,          # examples for log_generations (samples + entropy + lengths)
    eval_max_tokens: int = 1024,
    save_model: bool = False,
):
    config = dict(locals())
    run_dir = Path(output_dir) / run_name
    logger = LocalLogger(run_dir, config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------------- data ----------------
    r1_zero = load_prompt_template("r1_zero")
    train_examples = load_gsm8k_sft(train_path, r1_zero, n_examples=n_examples, seed=seed)
    val_pairs = build_prompts(load_gsm8k(val_path, limit=n_eval_examples), r1_zero)
    val_prompts = [p for p, _ in val_pairs]
    val_gts = [g for _, g in val_pairs]

    grad_accum_steps = batch_size // microbatch_size
    n_steps = math.ceil(len(train_examples) * n_epochs / batch_size)
    print(f"{len(train_examples)} train examples, {n_steps} optimizer steps, grad_accum={grad_accum_steps}")

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
    # With fp32 weights, run the forward in bf16 (speed/memory) while the optimizer updates fp32 weights.
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if param_dtype == "float32" else nullcontext()
    )

    llm = init_vllm(model_name, vllm_device, seed, vllm_gpu_memory_utilization)
    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=eval_max_tokens,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    warmup_steps = int(warmup_ratio * n_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------- eval helper ----------------
    def run_eval(step: int):
        t0 = time.time()
        load_policy_into_vllm_instance(policy, llm)
        metrics = evaluate_vllm(
            llm, eval_sampling_params, val_pairs, r1_zero_reward_fn,
            output_path=str(run_dir / "eval" / f"step_{step:05d}.jsonl"), batch_size=None,
        )
        gen = log_generations(
            llm, policy, tokenizer, val_prompts[:n_log_generations], val_gts[:n_log_generations],
            r1_zero_reward_fn, eval_sampling_params,
            output_path=str(run_dir / "generations.jsonl"), step=step,
        )
        logger.log("eval", step, {
            "accuracy": metrics["accuracy"],
            "format_rate": metrics["format_rate"],
            "mean_token_entropy": gen["summary"]["mean_token_entropy"],
            "response_length": gen["summary"]["response_length"],
            "response_length_correct": gen["summary"]["response_length_correct"],
            "response_length_incorrect": gen["summary"]["response_length_incorrect"],
            "eval_time_s": time.time() - t0,
        })
        logger.plot("eval")
        print(f"[eval step {step}] accuracy={metrics['accuracy']:.3f} format={metrics['format_rate']:.3f}")

    # ---------------- train loop ----------------
    run_eval(0)
    batches = iterate_sft_batches(train_examples, tokenizer, batch_size, microbatch_size, seed=seed)

    for step in range(1, n_steps + 1):
        t0 = time.time()
        microbatches = next(batches)
        step_loss = 0.0
        n_response_tokens = 0

        for mb in microbatches:
            input_ids = mb["input_ids"].to(policy_device)
            labels = mb["labels"].to(policy_device)
            response_mask = mb["response_mask"].to(policy_device)

            with autocast:
                out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            loss, _ = sft_microbatch_train_step(
                out["log_probs"].float(), response_mask, grad_accum_steps, normalize_constant
            )
            step_loss += loss.item()  # already divided by grad_accum_steps -> sums to the batch mean
            n_response_tokens += int(response_mask.sum())

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        logger.log("train", step, {
            "loss": step_loss,                                  # mean per-sequence NLL (sum over tokens)
            "loss_per_token": step_loss * batch_size * normalize_constant / max(1, n_response_tokens),
            "grad_norm": grad_norm.item(),                      # norm BEFORE clipping
            "lr": scheduler.get_last_lr()[0],
            "step_time_s": time.time() - t0,
        })
        if step % 10 == 0:
            logger.plot("train")
            print(f"[step {step}/{n_steps}] loss={step_loss:.3f} grad_norm={grad_norm.item():.2f}")

        if step % eval_every == 0 or step == n_steps:
            run_eval(step)

    logger.plot()
    if save_model:
        policy.save_pretrained(run_dir / "model")
        tokenizer.save_pretrained(run_dir / "model")


if __name__ == "__main__":
    typer.run(main)
