import os
import regex as re
from collections import Counter
from typing import Iterable, Iterator

# GPT-2 pre-tokenization pattern (with special-token handling done separately)
PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # id -> bytes 和 bytes -> id
        self.id_to_token = dict(vocab)
        self.token_to_id = {v: k for k, v in vocab.items()}

        # Merge the earlier pairs
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        self.special_set = set(self.special_tokens)
        if self.special_tokens:
            # Merge longer special tokens
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(tok) for tok in sorted_special]
            self.special_pattern = "|".join(escaped)
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls, 
        vocab_filepath : str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r") as f:
            vocab = {int(k): v.encode("utf-8") for k, v in (line.strip().split() for line in f)}

        with open(merges_filepath, "r") as f:
            merges = [(a.encode("utf-8"), b.encode("utf-8")) for a, b in (line.strip().split() for line in f)]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        # 1. Split by special tokens
        if self.special_pattern:
            pieces = re.split(f"({self.special_pattern})", text)
        else:
            pieces = [text]

        for piece in pieces:
            if not piece:
                continue
            if piece in self.special_set:
                ids.append(self.token_to_id[piece.encode("utf-8")])
                continue

            # 2. pre-tokenization
            for word in re.findall(PAT, piece):
                token_bytes_list = self._apply_bpe(word.encode("utf-8"))
                for tb in token_bytes_list:
                    ids.append(self.token_to_id[tb])

        return ids

    def _apply_bpe(self, token_bytes: bytes) -> list[bytes]:
        """Apply BPE to a sequence of bytes."""
        parts = [bytes([b]) for b in token_bytes]

        while len(parts) > 1:
            best_rank = None
            best_idx = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_idx = i
            if best_idx is None:
                break
            new_token = parts[best_idx] + parts[best_idx + 1]
            parts = parts[:best_idx] + [new_token] + parts[best_idx + 2:]

        return parts

    def decode(self, ids: Iterable[int]) -> str:
        byte_parts = []
        for i in ids:
            token = self.id_to_token.get(i)
            if token is None:
                raise ValueError(f"Unknown token id: {i}")
            byte_parts.append(token)
        all_bytes = b"".join(byte_parts)
        return all_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, text_iter: Iterator[str]) -> Iterator[list[int]]:
        buffer = ""
        for chunk in text_iter:
            buffer += chunk
            lines = buffer.split("\n")
            buffer = lines.pop()
            for line in lines:
                yield from self.encode(line + "\n")
        if buffer:
            yield from self.encode(buffer)