import json
from pathlib import Path
from typing import Callable

import torch
from vllm import LLM, SamplingParams

from cs336_alignment.sft.response_log_probs import get_response_log_probs
from cs336_alignment.sft.tokenize_prompt_output import tokenize_prompt_and_output


@torch.no_grad()
def _response_entropy_and_length(
    policy,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    batch_size: int,
) -> tuple[list[float], list[int]]:
    """Mean token entropy over response tokens (from the HF policy) and response length in tokens."""
    device = next(policy.parameters()).device
    entropies, lengths = [], []
    for start in range(0, len(prompts), batch_size):
        batch = tokenize_prompt_and_output(
            prompts[start:start + batch_size], responses[start:start + batch_size], tokenizer
        )
        mask = batch["response_mask"].to(device)
        out = get_response_log_probs(
            policy,
            batch["input_ids"].to(device),
            batch["labels"].to(device),
            return_token_entropy=True,
        )
        token_entropy = out["token_entropy"].to(device).float()
        n_tokens = mask.sum(dim=-1)
        mean_entropy = (token_entropy * mask).sum(dim=-1) / n_tokens.clamp(min=1)
        entropies.extend(mean_entropy.tolist())
        lengths.extend(n_tokens.tolist())
    return entropies, lengths


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def log_generations(
    vllm_model: LLM,
    policy,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params: SamplingParams,
    entropy_batch_size: int = 4,
    output_path: str | None = None,
    step: int | None = None,
) -> dict:
    """
    Generate responses with vLLM for the given prompts and log, per example:
    prompt, response, ground truth, rewards, mean token entropy, response length.
    Aggregates: mean rewards, mean entropy, mean response length (all / correct / incorrect).

    vllm_model should already hold the current policy weights (load_policy_into_vllm_instance).
    Entropy is computed with `policy` (HF model), since vLLM doesn't expose the full distribution.

    Returns {"summary": {...}, "examples": [...]}. If output_path is given, the examples are
    appended to it as jsonl (one line per example, tagged with `step`).
    """
    outputs = vllm_model.generate(prompts, sampling_params, use_tqdm=False)
    responses = [o.outputs[0].text for o in outputs]
    rewards = [reward_fn(r, gt) for r, gt in zip(responses, ground_truths)]

    was_training = policy.training
    policy.eval()
    entropies, lengths = _response_entropy_and_length(
        policy, tokenizer, prompts, responses, entropy_batch_size
    )
    if was_training:
        policy.train()

    examples = [
        {
            "step": step,
            "prompt": p,
            "response": r,
            "ground_truth": gt,
            "format_reward": rw["format_reward"],
            "answer_reward": rw["answer_reward"],
            "reward": rw["reward"],
            "mean_token_entropy": ent,
            "response_length": n,
        }
        for p, r, gt, rw, ent, n in zip(prompts, responses, ground_truths, rewards, entropies, lengths)
    ]

    correct = [e for e in examples if e["answer_reward"] == 1.0]
    incorrect = [e for e in examples if e["answer_reward"] != 1.0]
    summary = {
        "n": len(examples),
        "reward": _mean([e["reward"] for e in examples]),
        "format_reward": _mean([e["format_reward"] for e in examples]),
        "answer_reward": _mean([e["answer_reward"] for e in examples]),
        "mean_token_entropy": _mean(entropies),
        "response_length": _mean([e["response_length"] for e in examples]),
        "response_length_correct": _mean([e["response_length"] for e in correct]),
        "response_length_incorrect": _mean([e["response_length"] for e in incorrect]),
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as f:
            for e in examples:
                f.write(json.dumps(e) + "\n")

    return {"summary": summary, "examples": examples}
