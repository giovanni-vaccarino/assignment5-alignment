import torch

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None
) -> torch.Tensor:
    """
    Mask -> 1 represents include, 0 exclude.
    Tensor and mask have the same shape.
    If dim is None sum all over the dimensions, ow along the dimension provided.
    """
    masked_tensor = torch.where(mask == 1, tensor, 0.0)

    return torch.sum(masked_tensor, dim=dim) / normalize_constant
