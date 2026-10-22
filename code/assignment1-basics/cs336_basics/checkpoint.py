import numpy as np
import torch
import os
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int
from typing import BinaryIO

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike[str] | BinaryIO,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike[str] | BinaryIO,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(
        src,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


    