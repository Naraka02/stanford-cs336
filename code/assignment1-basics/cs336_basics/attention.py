import math

import torch
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange
from .utils import softmax
from .layers import Linear

class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        inv_freq = theta ** (
            -torch.arange(0, d_k, 2, device=device, dtype=torch.float32) 
            / d_k
        )
        positions = torch.arange(
            max_seq_len, device=device, dtype=torch.float32
        )

        angles = einsum(
            positions,
            inv_freq,
            "position, pair -> position pair",
        )

        self.register_buffer(
            "cos_cached", angles.cos(), persistent=False
        )
        self.register_buffer(
            "sin_cached", angles.sin(), persistent=False
        )

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        result = rearrange(
            [rotated_even, rotated_odd],
            "two ... pair -> ... (pair two)",
        )

        return result

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
):
    scores = einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys",
    ) / math.sqrt(Q.shape[-1])

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = softmax(scores, dim=-1)

    result = einsum(
        attention_weights, V,
        "... queries keys, ... keys d_v -> ... queries d_v",
    )

    return result


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = [
            rearrange(
                tensor,
                "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
                num_heads=self.num_heads,
            )
            for tensor in (q, k, v)
        ]

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            positions = token_positions.unsqueeze(-2)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        causal_mask = torch.ones(
            seq_len, seq_len, device=x.device, dtype=torch.bool
        ).tril()

        attention_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        attention_output = rearrange(
            attention_output,
            "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)",
        )

        result = self.output_proj(attention_output)

        return result