import torch

from collections.abc import Iterable

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.amax(dim=dim, keepdim=True)
    exp_x = (x - x_max).exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be non-negative")

    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm = torch.linalg.vector_norm(
        torch.stack([
            torch.linalg.vector_norm(grad)
            for grad in grads
        ])
    )

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(scale)
