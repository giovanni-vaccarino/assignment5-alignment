import json
import random
import re
from typing import Iterator

import torch

from cs336_alignment.sft.tokenize_prompt_output import tokenize_prompt_and_output

# GSM8K solutions contain calculator annotations like "48/2 = <<48/2=24>>24"; drop the <<...>> part.
_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


def format_gsm8k_response(answer_field: str) -> tuple[str, str]:
    """
    Turn a GSM8K "answer" field into an r1_zero-style response + short ground truth.

    The r1_zero prompt already ends with "Assistant: <think>", so the response continues from there:
        " <reasoning> </think> <answer> 72 </answer>"
    The exact "</think> <answer>" spacing matters: r1_zero_reward_fn checks for that literal string.
    """
    reasoning, final = answer_field.split("####")
    reasoning = _CALC_ANNOTATION.sub("", reasoning).strip()
    ground_truth = final.strip().replace(",", "")
    response = f" {reasoning} </think> <answer> {ground_truth} </answer>"
    return response, ground_truth


def load_gsm8k_sft(
    path: str,
    prompt_template: str,
    n_examples: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """
    Load GSM8K as SFT examples: [{"prompt", "response", "ground_truth"}].
    n_examples: take a random subset of this size (for the {128, 256, 512, 1024, full} sweep).
    """
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            response, ground_truth = format_gsm8k_response(ex["answer"])
            examples.append({
                "prompt": prompt_template.format(question=ex["question"]),
                "response": response,
                "ground_truth": ground_truth,
            })
    if n_examples is not None and n_examples < len(examples):
        examples = random.Random(seed).sample(examples, n_examples)
    return examples


def iterate_sft_batches(
    examples: list[dict],
    tokenizer,
    batch_size: int,
    microbatch_size: int,
    seed: int = 0,
) -> Iterator[list[dict[str, torch.Tensor]]]:
    """
    Endless iterator over optimizer steps. Each item is one batch (`batch_size` examples), already
    split into `batch_size // microbatch_size` tokenized microbatches (input_ids, labels, response_mask).

    Reshuffles every epoch and drops the last incomplete batch. Tokenizing per microbatch (not per
    batch) keeps padding to the longest sequence in each microbatch.
    """
    assert batch_size % microbatch_size == 0, "batch_size must be divisible by microbatch_size"
    assert batch_size <= len(examples), "batch_size larger than the dataset"
    rng = random.Random(seed)
    while True:
        order = list(range(len(examples)))
        rng.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch = [examples[i] for i in order[start:start + batch_size]]
            microbatches = []
            for m in range(0, batch_size, microbatch_size):
                mb = batch[m:m + microbatch_size]
                microbatches.append(tokenize_prompt_and_output(
                    [ex["prompt"] for ex in mb], [ex["response"] for ex in mb], tokenizer
                ))
            yield microbatches
