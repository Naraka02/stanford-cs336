import argparse
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import get_batch
from .loss import cross_entropy
from .model import TransformerLM
from .optimizer import AdamW
from .schedule import get_lr_cosine_schedule
from .utils import gradient_clipping

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    was_training = model.training
    model.eval()

    try:
        total_loss = 0.0
        for _ in range(num_batches):
            x, y = get_batch(
                dataset=dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()
        return total_loss / num_batches
    finally:
        model.train(was_training)

def load_tokens(path: str | Path, context_length: int) -> np.ndarray:
    tokens = np.load(path, mmap_mode="r", allow_pickle=False)

    if tokens.ndim != 1:
        raise ValueError(f"{path}: expected a one-dimensional token array")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f"{path}: token IDs must be integers")
    if len(tokens) <= context_length:
        raise ValueError(f"{path}: not enough tokens for one sample")

    return tokens

def train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_tokens(args.train_data, args.context_length)
    valid_data = load_tokens(args.valid_data, args.context_length)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    completed_steps = 0

    if args.resume is not None:
        completed_steps = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from checkpoint {args.resume} at step {completed_steps}")
    
    if not 0 <= completed_steps < args.steps:
        raise ValueError(f"resume checkpoint has invalid step {completed_steps}")

    model.train()

    for step in range(completed_steps, args.steps):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_steps,
            cosine_cycle_iters=args.steps,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=torch.device(args.device),
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = cross_entropy(logits, y)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss {loss.item()} at step {step + 1}")

        loss.backward()

        gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()
        completed_steps = step + 1

        if (
            completed_steps == 1
            or completed_steps % args.log_interval == 0
        ):
            print(
                f"step={completed_steps}/{args.steps} "
                f"train_loss={loss.item():.4f} "
                f"lr={lr:.6g}"
            )

        if (
            completed_steps % args.eval_interval == 0
            or completed_steps == args.steps
        ):
            valid_loss = evaluate(
                model=model,
                dataset=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                num_batches=args.eval_batches,
            )

            print(
                f"step={completed_steps}/{args.steps} "
                f"valid_loss={valid_loss:.4f}"
            )

        if (
            completed_steps % args.save_interval == 0
            or completed_steps == args.steps
        ):
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=completed_steps,
                out=output_dir / f"step_{completed_steps}.pt",
            )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--resume", default=None)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=352)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    positive_fields = (
        "vocab_size", "context_length", "d_model", "num_layers",
        "num_heads", "d_ff", "batch_size", "steps", "log_interval",
        "eval_interval", "eval_batches", "save_interval",
    )
    for name in positive_fields:
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")

    if args.d_model % args.num_heads != 0:
        parser.error("d_model must be divisible by num_heads")
    if (args.d_model // args.num_heads) % 2 != 0:
        parser.error("RoPE requires an even head dimension")
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("warmup_steps must satisfy 0 <= warmup_steps < steps")
    if not 0 <= args.min_lr <= args.max_lr:
        parser.error("learning rates must satisfy 0 <= min_lr <= max_lr")
    if args.grad_clip <= 0:
        parser.error("grad_clip must be positive")

    return args

if __name__ == "__main__":
    train(parse_args())