import torch
from typing import Literal
from cs336_alignment.grpo.masked_mean import masked_mean
from cs336_alignment.grpo.pg_wrapper import compute_policy_gradient_loss

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward (not really??) and backward pass on a microbatch
    """
    # 1. Compute policy gradient loss per token
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    # 2. Aggregate loss and call backward
    loss = masked_mean(per_token_loss, response_mask, dim=None) / gradient_accumulation_steps
    loss.backward()

    return loss, metadata
    
 