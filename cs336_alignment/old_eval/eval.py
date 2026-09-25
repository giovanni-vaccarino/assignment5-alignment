from typing import Callable, List
import json
from pathlib import Path
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams

def eval(model_name: str, test_set_path: str, output_path: str):
    # 1. Load jsnol test set and convert to prompts"
    prompts = []
    ground_truths = []
    with open(test_set_path, "r") as f:
        for row in f:
            data = json.loads(row)
            prompts.append(build_prompt(data.get("problem", "")))
            ground_truths.append(data.get("answer", ""))
    
    llm = LLM(model=model_name)

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # 2. Call evalute vllm method
    evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        ground_truths,
        eval_sampling_params,
        output_path
    )

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for output, ground_truth in zip(outputs, ground_truths):
        generated_text = "<think>" + output.outputs[0].text
        reward_result = reward_fn(generated_text, ground_truth)
        results.append({
            "question": output.prompt,
            "response": generated_text,
            "ground_truth": ground_truth,
            "reward": reward_result.get("reward", 0.0),
            "format_reward": reward_result.get("format_reward", 0.0),
            "answer_reward": reward_result.get("answer_reward", 0.0)
        })

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Aggregate metrics
    n = len(results)
    avg_reward = sum(r["reward"] for r in results) / n
    avg_format = sum(r["format_reward"] for r in results) / n
    avg_answer = sum(r["answer_reward"] for r in results) / n
    print(f"N={n} | avg_reward={avg_reward:.4f} | "
          f"avg_format={avg_format:.4f} | avg_answer={avg_answer:.4f}")

def build_prompt(question):
    template = Path("cs336_alignment/prompts/r1_zero.prompt").read_text(encoding="utf-8")

    return template.format(question=question)
