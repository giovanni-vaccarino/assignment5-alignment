import torch

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs)["input_ids"]
    output_ids = tokenizer(output_strs)["input_ids"]

    input_ids = []
    masks = []
    max_seq_len = 0
    for prompt, output in zip(prompt_ids, output_ids):
        input_ids.append(prompt + output)
        masks.append([False]*len(prompt) + [True]*len(output))
        max_seq_len = max(max_seq_len, len(prompt) + len(output))

    for seq, mask in zip(input_ids, masks):
        pad_len = max_seq_len - len(seq)
        seq += pad_len * [tokenizer.pad_token_id]
        mask += pad_len * [False]

    input_ids = torch.tensor(input_ids)
    response_mask = torch.tensor(masks)

    return {
        "input_ids": input_ids[..., :-1],
        "labels": input_ids[..., 1:],
        "response_mask": response_mask[..., 1:]
    }
