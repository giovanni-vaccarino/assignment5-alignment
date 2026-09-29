import torch

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor
) -> torch.Tensor:
    """
    Naive Policy gradient loss is - A * log p

    raw_rewards_or_advantages.shape -> (batch_size, 1)
    policy_log_probs.shape -> (batch_size, sequence_length)

    returns (batch_size, sequence_length)
    """

    return - raw_rewards_or_advantages * policy_log_probs
