import json
from collections import Counter
from pathlib import Path
from typing import Callable, List, Tuple

import typer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# Dr. GRPO grader returns something like
# {
#     "format_reward": 1.0,
#     "answer_reward": 0.0,
#     "reward": 0.0
# }

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_gsm8k(path: str, limit: int | None = None) -> List[dict]:
    """Load GSM8K jsonl into [{"question", "ground_truth"}], ground truth = number after '####'."""
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            ground_truth = ex["answer"].split("####")[-1].strip().replace(",", "")
            examples.append({"question": ex["question"], "ground_truth": ground_truth})
            if limit is not None and len(examples) >= limit:
                break
    return examples


def load_prompt_template(name: str = "r1_zero") -> str:
    return (PROMPTS_DIR / f"{name}.prompt").read_text()


def build_prompts(examples: List[dict], template: str) -> List[Tuple[str, str]]:
    return [(template.format(question=ex["question"]), ex["ground_truth"]) for ex in examples]


def summarize(results: List[dict]) -> dict:
    """Counts for the three (format, answer) categories + accuracy."""
    categories = Counter(
        (r["rewards"]["format_reward"], r["rewards"]["answer_reward"]) for r in results
    )
    n = len(results)
    return {
        "n": n,
        "accuracy": sum(r["rewards"]["answer_reward"] for r in results) / max(n, 1),
        "format_rate": sum(r["rewards"]["format_reward"] for r in results) / max(n, 1),
        "format1_answer1": categories[(1.0, 1.0)],
        "format1_answer0": categories[(1.0, 0.0)],
        "format0_answer0": categories[(0.0, 0.0)],
    }


def evaluate_vllm(
    vllm_model: LLM,
    eval_sampling_params: SamplingParams,
    prompts: List[Tuple[str, str]],
    reward_fn: Callable[[str, str], dict[str, float]],
    output_path: str | None = None,
) -> dict:
    """
    Evaluate a model on a list of prompts, compute eval metrics and serialize to disk.
    """
    results = []

    # Serialized version (batch size = 1)
    for prompt, ground_truth in prompts:
        outputs = vllm_model.generate(prompt, eval_sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        rewards = reward_fn(generated_text, ground_truth)
        results.append({
            "prompt": prompt,
            "response": generated_text,
            "ground_truth": ground_truth,
            "rewards": rewards,
        })

    metrics = summarize(results)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def main(
    data_path: str = "data/gsm8k/test.jsonl",
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    output_path: str = "outputs/eval/zero_shot_gsm8k.jsonl",
    limit: int | None = None,
    max_tokens: int = 1024,
):
    examples = load_gsm8k(data_path, limit=limit)
    prompts = build_prompts(examples, load_prompt_template("r1_zero"))

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        swap_space=0,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
        # Without this "</answer>" is stripped and r1_zero_reward_fn always gives format_reward 0.
        include_stop_str_in_output=True,
    )

    metrics = evaluate_vllm(llm, sampling_params, prompts, r1_zero_reward_fn, output_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    typer.run(main)
