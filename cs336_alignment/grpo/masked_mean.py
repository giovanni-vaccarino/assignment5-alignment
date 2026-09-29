import torch

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
    """
    Following the mask average the elements of the tensor along the dim (if provided, ow all elements).
    """

    # Apply mask
    tensor = torch.where(mask == 1, tensor, 0.0)
    num_elements = torch.count_nonzero(mask, dim=dim)

    # Do the mean operation
    return torch.sum(tensor, dim=dim) / num_elements
