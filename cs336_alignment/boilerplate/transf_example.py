from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

tokenizer = AutoTokenizer("Qwen/Qwen2.5-Math-1.5B")

input_ids = torch.tensor([])
labels = torch.tensor([])

logits = model(input_ids).logits()
loss = F.cross_entropy(logits, labels)

model.save_pretrained(save_directory=".")
tokenizer.save_pretrained(save_directory=".")

# Gradient accumulation

data_loader = [()]
optimizer : torch.optim.optimizer = 0
gradient_accumulation_steps = 8

for idx, (inputs, labels) in enumerate(data_loader):
    logits = model(inputs).logits
    loss = F.cross_entropy(logits, labels) / gradient_accumulation_steps

    loss.backward()

    if (idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()