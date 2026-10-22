import math 

import torch
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int

from .model import TransformerLM
from .utils import softmax

def sample_next_token(
    logits: Float[Tensor, "batch_size vocab_size"],
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Int[Tensor, "batch_size"]:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab_size]")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and >= 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.float()
    probs = softmax(scores / temperature, dim=-1)

    if top_p == 1.0:
        return torch.multinomial(probs, num_samples=1)

    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    filtered_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

    sampled_ids = torch.multinomial(filtered_probs, num_samples=1)

    return sorted_ids.gather(dim=-1, index=sampled_ids)

@torch.inference_mode()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if model.context_length <= 0:
        raise ValueError("model.context_length must be positive")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and >= 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    device = next(model.parameters()).device

    tokens = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            context = tokens[:, -model.context_length:]

            logits = model(context)

            next_token = sample_next_token(
                logits=logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
            )

            tokens = torch.cat((tokens, next_token), dim=-1)

            if (
                eos_token_id is not None
                and next_token.item() == eos_token_id
            ):
                break

        return tokens.squeeze(0).tolist()
    finally:
        model.train(was_training)