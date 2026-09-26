import torch
from cs336_alignment.sft.masked_normalize import masked_normalize

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    policy_log_probs -> (B, T)
    response_mask    -> (B, T)

    return:
        loss
        metadata
    """
    loss = - masked_normalize(policy_log_probs, response_mask, normalize_constant) / (gradient_accumulation_steps * policy_log_probs.shape[0])
    loss.backward()

    metadata = {}

    return loss, metadata
