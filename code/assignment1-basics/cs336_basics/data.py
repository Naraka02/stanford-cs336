import torch
from torch import nn, Tensor
import numpy as np
from jaxtyping import Bool, Float, Int

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | None = None,
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    if dataset.ndim != 1:
        raise ValueError("dataset must be a one-dimensional token array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain at least context_length + 1 tokens")

    starts = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=batch_size,
    )

    sequences = np.stack([
        dataset[start : start + context_length + 1]
        for start in starts
    ])

    tokens = torch.tensor(sequences, dtype=torch.long, device=device)

    x = tokens[:, :-1].contiguous()
    y = tokens[:, 1:].contiguous()

    return x, y