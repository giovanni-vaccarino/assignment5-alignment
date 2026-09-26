import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Given the logits, we compute and return the entropy.
    logits.shape -> (..., V)
    H(p) = - sum(p(x)*log(p(x))) ; H.shape -> (...)
    """
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    exponent = logits - lse
    p_logp = torch.where(torch.exp(exponent) > 0, exponent * torch.exp(exponent), 0.0)
    return -torch.sum((p_logp), dim=-1)
