#!/usr/bin/env python3
"""Lane mx: P1's 27B short golden (intarsia-golden-topk-v1) plus synthetic long prompts, so `accuracy --mode prefill
--write-self-golden` at --prefill-chunk 256 and 4096 meets every tail case (after lane ck's make_synth.py).

    make_tails.py IN.safetensors OUT.safetensors IDS.npy N[,N...]

IDS.npy: int32 token ids of a long text, tokenized with the 27B's own tokenizer (scripts/mx/tokenize_ids.py); each prompt
synth_<N> is the first N of them (cycled). Top-k tensors are placeholders (only the engine's own dumps are compared).
"""
import json, struct, sys
import numpy as np


def read(path):
    b = open(path, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    return json.loads(b[8:8 + n]), b[8 + n:]


def main():
    src, out, idsf = sys.argv[1], sys.argv[2], sys.argv[3]
    Ns = [int(x) for x in sys.argv[4].split(",")]
    base = np.load(idsf).astype(np.int32)
    h, data = read(src)
    meta = h.pop("__metadata__")
    prompts = json.loads(meta["prompts"])
    k = int(meta.get("topk", 20))
    add = []
    for N in Ns:
        syn = np.tile(base, N // len(base) + 1)[:N].astype(np.int32)
        j = len(prompts)
        prompts.append({"name": f"synth_{N}", "kind": "raw", "n_tokens": N, "n_prompt": N, "continuation": 0, "text": ""})
        add += [(f"ids_{j}", "I32", [N], syn.tobytes()),
                (f"topk_ids_{j}", "I32", [N, k], np.zeros((N, k), np.int32).tobytes()),
                (f"topk_logprobs_{j}", "F32", [N, k], np.full((N, k), -3.0, np.float32).tobytes())]
    meta["prompts"] = json.dumps(prompts)
    meta["synth_note"] = "synth_*: ids of a long text (27B tokenizer); top-k tensors are placeholders (lane mx)"
    header, off, blob = {"__metadata__": meta}, 0, []
    for name, v in h.items():
        raw = data[v["data_offsets"][0]:v["data_offsets"][1]]
        header[name] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + len(raw)]}
        blob.append(raw); off += len(raw)
    for name, dt, shape, raw in add:
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blob.append(raw); off += len(raw)
    hj = json.dumps(header).encode()
    hj += b" " * ((8 - (8 + len(hj)) % 8) % 8)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        for r in blob:
            f.write(r)
    print(f"{out}: {len(prompts)} prompts, added {Ns}")


main()
