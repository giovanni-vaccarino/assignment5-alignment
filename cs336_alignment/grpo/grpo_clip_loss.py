import torch

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    - min (ratio * A, clip(1 + eps, 1 - eps, ratio) * A)

    advantages.shape        -> (batch_size, 1)
    policy_log_probs.shape  -> (batch_size, seq_length)
    old_log_probs.shape     -> (batch_size, seq_length)

    returns (batch_size, seq_length)
    """
    assert policy_log_probs.shape == old_log_probs.shape

    ratio = torch.exp(policy_log_probs - old_log_probs) # (batch_size, seq_length)
    clipped_ratio = torch.clip(ratio, 1 - cliprange, 1 + cliprange) # (batch_size, seq_length)
    combined_ratios = torch.stack((ratio * advantages, clipped_ratio * advantages), dim=-1) # (batch_size, seq_length, 2)
    loss = - torch.min(combined_ratios, dim=-1).values # (batch_size, seq_length)

    metadata = {"clipped": clipped_ratio * advantages < ratio * advantages}   # (B, T) bool

    return loss, metadata
