"""DeepSeek-V4.1 Engram hash parameters and a hash fixture for `intarsia_core::dsv41::engram` (lane 1a).

Mirrors the official inference/engram.py (MIT, DeepSeek): EngramLayout.from_args (primes), compute_hash_multipliers
(numpy PCG64, which is why this is a script and not Rust) and NgramHashState.forward (the rolling XOR hash). Only numpy is
needed (sympy.isprime is replaced by trial division, exact for these sizes). Writes JSON to stdout:

    uv run --no-project --with numpy python tools/gen_dsv41_engram_params.py \
        > crates/intarsia-core/src/dsv41/engram_fixture.json
"""
import json

import numpy as np

# config.json (text_config) of DeepSeek-V4.1-Flash-1017
LAYER_IDS = (1, 14)
MAX_NGRAM = 4
N_HEADS = 8
VOCAB = 16_000_000
COMPRESSED_VOCAB = 99_092
NUM_EMBEDDINGS = (384_006_168, 384_016_682)


def isprime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def find_next_prime(start, seen):
    c = start + 1
    while not isprime(c) or c in seen:
        c += 1
    return c


def layout():
    primes, seen = [], set()
    for _ in LAYER_IDS:
        per_ngram = []
        for _ in range(MAX_NGRAM - 1):
            sizes, cur = [], VOCAB - 1
            for _ in range(N_HEADS):
                cur = find_next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


def multipliers():
    mx = np.iinfo(np.int64).max
    bound = max(1, (mx // COMPRESSED_VOCAB) // 2)
    rows = []
    for layer in LAYER_IDS:
        g = np.random.default_rng(10007 * layer)
        rows.append([int(v) * 2 + 1 for v in g.integers(low=0, high=bound, size=(MAX_NGRAM,), dtype=np.int64)])
    return rows


def hashes(primes, mults, compressed, pad_id, dead=-1):
    """NgramHashState.forward for one sequence prefilled from position 0: [L, n_layers, (MAX_NGRAM - 1) * N_HEADS]."""
    flat = [[p for per in layer for p in per] for layer in primes]
    offsets = [list(np.cumsum([0, *f[:-1]])) for f in flat]
    out = []
    for pos in range(len(compressed)):
        toks, blocked = [], False
        for shift in range(MAX_NGRAM):
            src = compressed[max(pos - shift, 0)]
            blocked = blocked or pos < shift or src == dead
            toks.append(pad_id if blocked else src)
        row = []
        for li in range(len(LAYER_IDS)):
            prod = [t * m for t, m in zip(toks, mults[li])]
            rolling, cols = prod[0], []
            for i in range(1, MAX_NGRAM):
                rolling ^= prod[i]
                cols.extend(rolling % p for p in primes[li][i - 1])
            row.append([int(c + o) for c, o in zip(cols, offsets[li])])
        out.append(row)
    return out


if __name__ == "__main__":
    primes = layout()
    for li, n in enumerate(NUM_EMBEDDINGS):
        assert sum(p for per in primes[li] for p in per) == n, (li, n)
    mults = multipliers()
    seq = [5, 17, 99_091, 3, 42, -1, 7, 7, 0]
    print(json.dumps({"layer_ids": LAYER_IDS, "primes": primes, "multipliers": mults, "pad_id": 2, "compressed": seq,
                      "hashes": hashes(primes, mults, seq, 2)}))
