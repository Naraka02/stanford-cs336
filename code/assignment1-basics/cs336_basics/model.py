import math

import torch
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange
from .layers import Linear, RMSNorm, SwiGLU, Embedding
from .attention import RoPE, MultiheadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        rope = RoPE(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device,
        )

        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        x = x + self.attn(
            self.ln1(x),
            token_positions=token_positions,
        )
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        token_ids: Int[Tensor, " batch sequence_length"],
        token_positions: Int[Tensor, " batch sequence_length"] | None = None,
    ) -> Float[Tensor, " batch sequence_length vocab_size"]:
        seq_len = token_ids.shape[-1]

        if seq_len > self.context_length:
            raise ValueError("equence length exceeds context_length")

        if token_positions is None:
            token_positions = torch.arange(
                seq_len,
                device=token_ids.device,
            )

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits