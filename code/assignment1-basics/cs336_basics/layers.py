import math

import torch
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum, reduce
from .utils import silu

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0*std,
            b=3.0*std,
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: Int[Tensor, " ..."]) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square = reduce(x.square(), " ... d_model -> ... 1", "mean")
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        result = x * inverse_rms * self.weight

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        gate = silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)