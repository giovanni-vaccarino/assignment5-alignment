from typing import Callable
import torch
from einops import rearrange

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalized_by_std: bool
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    1. Compute reward values
    2. Compute advantages
    """

    unnormalized_rewards = []
    format_rewards = []

    for rollout, ground_truth in zip(rollout_responses, repeated_ground_truths):
        rewards = reward_fn(rollout, ground_truth)
        # we may save format and answer reward for logging
        unnormalized_rewards.append(rewards.get("reward", 0))
        format_rewards.append(rewards.get("format_reward", 0))

    rollout_rewards = rearrange(torch.tensor(unnormalized_rewards), "(B G) -> B G", G=group_size)

    advantages = rollout_rewards - torch.mean(rollout_rewards, dim=-1, keepdim=True)
    if normalized_by_std: # Standard GRPO implementation
        advantages = advantages / (torch.std(rollout_rewards, dim=-1, keepdim=True) + advantage_eps)

    advantages = rearrange(advantages, "B G -> (B G)")

    metadata = {
        "format_rewards": format_rewards
    }

    return advantages, torch.tensor(unnormalized_rewards), metadata


    