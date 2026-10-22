import math

import torch
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int

def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, "..."]:
    logits = logits - logits.amax(dim=-1, keepdim=True)

    # log(sum(exp(logits)))
    log_normalizer = logits.exp().sum(dim=-1).log()

    target_logits = logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    return (log_normalizer - target_logits).mean()