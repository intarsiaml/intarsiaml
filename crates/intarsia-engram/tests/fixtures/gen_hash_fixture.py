"""Generate the Engram hash parity fixture from DeepSeek's official `inference/engram.py` (MIT).

Runs the official `NgramHashState.forward` on CPU and writes `hash_dsv41.json`, which
`tests/hash_parity.rs` checks the Rust hasher against bit for bit.

Usage (any env with torch, numpy, sympy, tokenizers):
    python gen_hash_fixture.py --official /path/to/DeepSeek-V4.1-Flash/inference \
        --tokenizer /path/to/tokenizer.json --out hash_dsv41.json [--map-out token_map.json]

Inputs are generated, not stored: a splitmix64 stream (seeded per sequence) that the Rust test
replays. Sequences are in *compressed* id space (the official token map is swapped for the
identity after init, so pad stays `token_map[2]`); the token map itself is checked separately
by sha256 (tests/hash_parity.rs, gated on INTARSIA_DSV41_TOKENIZER) and by a few real-text
sequences hashed through the full official pipeline (raw ids -> real map -> hash).
"""
import argparse
import hashlib
import json
import sys
import types

import numpy as np
import torch

MASK64 = (1 << 64) - 1
N_SEQ = 3000
SEED = 0x5EED_E46A


def splitmix(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def gen_sequence(k, vocab, pad):
    """Sequence k: (compressed ids, dead mask or None). Mirrored in tests/hash_parity.rs."""
    s = (SEED ^ (k * 0xD1B54A32D192ED03)) & MASK64
    s, r = splitmix(s)
    n = 1 + r % 64
    ids, dead = [], []
    with_dead = k % 7 == 3
    for _ in range(n):
        s, r = splitmix(s)
        kind = r % 16
        if kind == 0:
            t = pad
        elif kind == 1:
            t = (r >> 8) % 64
        elif kind == 2:
            t = vocab - 1 - (r >> 8) % 4
        else:
            t = (r >> 8) % vocab
        ids.append(t)
        s, r = splitmix(s)
        dead.append(with_dead and r % 5 == 0)
    return ids, (dead if with_dead else None)


def fnv1a(values):
    h = 0xCBF29CE484222325
    for v in values:
        for b in int(v).to_bytes(8, "little", signed=False):
            h ^= b
            h = (h * 0x100000001B3) & MASK64
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-out")
    a = ap.parse_args()
    sys.path.insert(0, a.official)
    import engram  # official DeepSeek module
    from tokenizers import Tokenizer

    cfg = json.load(open(f"{a.official}/config.json"))
    args = types.SimpleNamespace(**cfg, max_batch_size=4, max_seq_len=256)

    tok = Tokenizer.from_file(a.tokenizer)

    class Shim:  # what build_compressed_token_map uses of a HF fast tokenizer
        backend_tokenizer = tok

        def __len__(self):
            return tok.get_vocab_size(with_added_tokens=True)

    layout = engram.EngramLayout.from_args(args)
    state = engram.NgramHashState(args, layout, Shim())
    real_map = state.token_map.clone()
    vocab = args.engram_compressed_vocab_size
    pad = int(state.pad_id)
    map_bytes = np.asarray(real_map.numpy(), dtype="<u4").tobytes()
    map_sha = hashlib.sha256(map_bytes).hexdigest()
    if a.map_out:
        json.dump(real_map.tolist(), open(a.map_out, "w"))

    def run(ids_rows, start_pos=0, mask=None):
        t = torch.tensor(ids_rows, dtype=torch.int64)
        m = None if mask is None else torch.tensor(mask, dtype=torch.bool)
        return state(t, start_pos, m)  # [B, L, layers, 24]

    # real text through the full official pipeline (raw ids -> real map)
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "THE QUICK BROWN FOX jumps over the LAZY dog!",
        "fn main() {\n    println!(\"hello, world\");\n}\n",
        "Engram: conditional memory via scalable lookup. Ünïcödé façade naïve café.",
        "  leading spaces\t\ttabs\r\nand CRLF  ",
        "数据并行与张量并行在大模型推理中的区别是什么？",
        "def f(x):\n    return x ** 2  # square\n",
        "1234567890 3.14159 e^{i\\pi} + 1 = 0",
    ]
    real = []
    for text in texts:
        raw = tok.encode(text, add_special_tokens=False).ids
        out = run([raw])[0].reshape(-1).tolist()
        real.append({"text": text, "raw_ids": raw, "digest": f"{fnv1a(out):016x}"})

    # identity map: inputs are compressed ids from here on
    state.token_map = torch.arange(vocab, dtype=torch.int64)

    seqs, full = [], []
    for k in range(N_SEQ):
        ids, dead = gen_sequence(k, vocab, pad)
        mask = None if dead is None else [[not d for d in dead]]
        out = run([ids], 0, mask)[0]
        flat = out.reshape(-1).tolist()
        if k % 5 == 0 and len(ids) > 1:  # prefill/decode split must agree with one pass
            cut = len(ids) // 2
            m = None if dead is None else [[not d for d in dead]]
            parts = [run([ids[:cut]], 0, None if m is None else [m[0][:cut]])[0]]
            for p in range(cut, len(ids)):
                parts.append(run([ids[p:p + 1]], p, None if m is None else [m[0][p:p + 1]])[0])
            assert torch.equal(torch.cat(parts, 0), out), k
        if k % 11 == 0 and dead is None:  # batch > 1 must agree with batch 1
            other, _ = gen_sequence(k + 100000, vocab, pad)
            other = (other * 64)[: len(ids)]
            b = run([ids, other], 0)
            assert torch.equal(b[0], out), k
        seqs.append(f"{fnv1a(flat):016x}")
        if k < 4:
            full.append({"k": k, "rows": out.tolist()})

    fixture = {
        "source": "DeepSeek-V4.1-Flash inference/engram.py NgramHashState (MIT), run on CPU by gen_hash_fixture.py",
        "seed": SEED,
        "n_seq": N_SEQ,
        "compressed_vocab": vocab,
        "pad": pad,
        "layer_ids": list(layout.layer_ids),
        "num_embeddings": list(layout.num_embeddings),
        "primes": [[list(h) for h in layer] for layer in layout.primes],
        "multipliers": state.multipliers.tolist(),
        "offsets": state.offsets.tolist(),
        "token_map_len": len(real_map),
        "token_map_sha256_u32le": map_sha,
        "digest": "fnv1a64 over row ids as u64 LE, order [position][layer][column]",
        "digests": seqs,
        "full": full,
        "real_text": real,
    }
    json.dump(fixture, open(a.out, "w"), separators=(",", ":"))
    print("token map sha256(u32le):", map_sha, "pad", pad, "vocab", vocab, "len", len(real_map))


if __name__ == "__main__":
    main()
