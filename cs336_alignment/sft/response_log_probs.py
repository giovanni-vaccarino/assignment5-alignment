import torch
from cs336_alignment.sft.per_token_entropy import compute_entropy

def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits # (B, T, V)

    lse = torch.logsumexp(logits, dim=-1)
    log_probs = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) - lse

    return_dict = {"log_probs": log_probs}
    if return_token_entropy:
        return_dict["token_entropy"] = compute_entropy(logits)

    return return_dict
