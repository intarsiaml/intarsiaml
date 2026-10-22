#!/usr/bin/env python3
"""fx1: make the oracle prompt set: every prompt of golden A plus the named prompts of golden B, as one
intarsia-golden-topk-v1 file (prefill rows only: ids_N, topk_ids_N, topk_logprobs_N; decode tensors are dropped).

usage: merge_goldens.py A.safetensors B.safetensors NAME[,NAME...] OUT.safetensors

The metadata is A's (engine, model, template, kv_dtype, ...) with `prompts` extended by B's entries and a
`merged_from` note naming both files and their sha256. Used to build
golden/oracle/<model>-oracle-set source goldens: the CPU oracle is then made from OUT with
`intarsia accuracy --backend cpu --golden OUT --mode prefill --write-self-golden ...`.
"""
import hashlib
import json
import struct
import sys


def read(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        data = f.read()
    return h, data


def tensor(h, data, name):
    e = h[name]
    b, en = e["data_offsets"]
    return e["dtype"], e["shape"], data[b:en]


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    a, b, names, out = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
    ha, da = read(a)
    hb, db = read(b)
    pa = json.loads(ha["__metadata__"]["prompts"])
    pb = json.loads(hb["__metadata__"]["prompts"])
    tensors = []
    prompts = []
    for i, p in enumerate(pa):
        prompts.append(p)
        tensors += [(f"{k}_{len(prompts) - 1}",) + tensor(ha, da, f"{k}_{i}") for k in ("ids", "topk_ids", "topk_logprobs")]
    for nm in names:
        j = next((j for j, p in enumerate(pb) if p["name"] == nm), None)
        if j is None:
            sys.exit(f"{b}: no prompt {nm}")
        prompts.append(pb[j])
        tensors += [(f"{k}_{len(prompts) - 1}",) + tensor(hb, db, f"{k}_{j}") for k in ("ids", "topk_ids", "topk_logprobs")]
    ka, kb = ha["__metadata__"].get("topk"), hb["__metadata__"].get("topk")
    if ka != kb:
        sys.exit(f"top-k differs: {ka} vs {kb}")
    meta = dict(ha["__metadata__"])
    meta["prompts"] = json.dumps(prompts)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    meta["merged_from"] = json.dumps({"all_of": a.split("/")[-1], "all_of_sha256": sha(a), "prompts_of": b.split("/")[-1], "prompts_of_sha256": sha(b), "prompts": names})
    header, off, blob = {"__metadata__": meta}, 0, []
    for name, dt, shape, raw in tensors:
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blob.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    hj += b" " * ((8 - (8 + len(hj)) % 8) % 8)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for r in blob:
            f.write(r)
    print(f"{out}: {len(prompts)} prompts, {sum(p.get('n_tokens', 0) for p in prompts)} tokens")


if __name__ == "__main__":
    main()
