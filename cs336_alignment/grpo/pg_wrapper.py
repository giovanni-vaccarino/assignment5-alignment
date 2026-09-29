import torch
from typing import Literal
from cs336_alignment.grpo.naive_pg_loss import compute_naive_policy_gradient_loss
from cs336_alignment.grpo.grpo_clip_loss import compute_grpo_clip_loss

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    switch(loss_type):
        case "no_baseline":
            naive_pg_loss
        case "reinforce_with_baseline":
            naive with group normalization
        case "grpo_clip":
            GRPO clip loss
    """

    metadata = {}
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    else:
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    return loss, metadata

