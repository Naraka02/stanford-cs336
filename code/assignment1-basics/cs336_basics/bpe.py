import os
import regex as re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import BinaryIO

# GPT-2 pre-tokenization pattern (with special-token handling done separately)
PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _pretokenize_chunk(args):
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    # Split on special tokens first so they aren't merged into normal text
    if special_tokens:
        # Escape and join
        pattern = "|".join(re.escape(t) for t in special_tokens)
        segments = re.split(f"({pattern})", text)
    else:
        segments = [text]

    counter: Counter[tuple[bytes, ...]] = Counter()
    specials = set(special_tokens)
    for seg in segments:
        if not seg:
            continue
        if seg in specials:
            continue  # special tokens handled in vocab, not merged
        for m in PAT.finditer(seg):
            token = m.group().encode("utf-8")
            counter[tuple(bytes([b]) for b in token)] += 1
    return counter

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. Initialize vocabulary
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        if tok.encode("utf-8") not in vocab.values():
            vocab[next_id] = tok.encode("utf-8")
            next_id += 1

    # 2. Pre-tokenization
    num_processes = 4
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [
        (str(input_path), s, e, special_tokens)
        for s, e in zip(boundaries[:-1], boundaries[1:])
        if e > s
    ]

    word_counts: Counter[tuple[bytes, ...]] = Counter()
    if len(tasks) <= 1:
        for t in tasks:
            word_counts.update(_pretokenize_chunk(t))
    else:
        with ProcessPoolExecutor(
            max_workers=num_processes,
            mp_context=get_context("spawn"),
        ) as ex:
            for c in ex.map(_pretokenize_chunk, tasks):
                word_counts.update(c)

    # Represent each unique word as a list of token ids (initially byte ids 0..255)
    # and keep its frequency.
    # word -> list[int] of current token ids
    words: dict[tuple[int, ...], int] = {}
    for word, freq in word_counts.items():
        ids = tuple(b[0] for b in word)  # bytes([i]) -> id i
        words[ids] = words.get(ids, 0) + freq

    merges: list[tuple[bytes, bytes]] = []

    # 3. Build pair counts and an index of the words containing each pair.
    # Word IDs stay fixed; their frequencies remain valid after every merge.
    word_tokens = list(words)
    word_frequencies = list(words.values())
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)

    for word_id, ids in enumerate(word_tokens):
        freq = word_frequencies[word_id]
        for pair, occurrences in Counter(zip(ids, ids[1:])).items():
            pair_counts[pair] += occurrences * freq
            pair_words[pair].add(word_id)

    while next_id < vocab_size and pair_counts:
        # Preserve the reference implementation's byte-based tie-break.
        best = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]),
        )
        a, b = best
        new_id = next_id
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))
        next_id += 1

        # Snapshot the set because the index is updated during the merge.
        affected_words = tuple(pair_words[best])
        for word_id in affected_words:
            ids = word_tokens[word_id]
            freq = word_frequencies[word_id]
            old_pairs = Counter(zip(ids, ids[1:]))

            # Merge left to right, without overlapping occurrences.
            out: list[int] = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1

            new_ids = tuple(out)
            word_tokens[word_id] = new_ids
            new_pairs = Counter(zip(new_ids, new_ids[1:]))

            # Only pairs whose multiplicity changed need count updates.
            for pair in old_pairs.keys() | new_pairs.keys():
                old_count = old_pairs[pair]
                new_count = new_pairs[pair]
                if old_count == new_count:
                    continue

                pair_counts[pair] += (new_count - old_count) * freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                if new_count == 0:
                    pair_words[pair].remove(word_id)
                    if not pair_words[pair]:
                        del pair_words[pair]
                elif old_count == 0:
                    pair_words[pair].add(word_id)

    return vocab, merges
