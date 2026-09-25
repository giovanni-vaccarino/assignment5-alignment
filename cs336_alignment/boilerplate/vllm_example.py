from vllm import LLM, SamplingParams

prompts = [
    "2 + 2 = ",
]

sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["\n"]
)

model_name = "Qwen/Qwen2.5-Math-1.5B"

# Settings for a 6GB laptop GPU / 16GB RAM machine:
# swap_space=0 avoids pinning 4GiB of host RAM, enforce_eager skips CUDA graph capture.
llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.8,
    max_model_len=2048,
    swap_space=0,
    enforce_eager=True,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
