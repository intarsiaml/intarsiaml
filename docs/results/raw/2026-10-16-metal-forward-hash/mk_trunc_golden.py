#!/usr/bin/env python3
"""mk_trunc_golden.py SRC DST: prefix-truncated top-k golden. The fp32 reference is one causal full-sequence forward, so
row i of a prompt's first n tokens is exactly row i of the full golden: ids_k[:n] and topk_*_k[:n] are valid references
for the truncated prompt. Stdlib only. Prompts: (source prompt index, n) in CUTS."""
import json, struct, sys
CUTS = [(0, 257), (1, 513), (0, 258), (1, 769), (0, 2049)]  # lengths 1 mod 256 (257, 513, 769, 2049) and a control (258)
src, dst = sys.argv[1:3]
f = open(src, "rb")
n = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(n))
base = 8 + n
data = open(src, "rb").read()
meta = dict(hdr["__metadata__"])
prompts = json.loads(meta["prompts"])
def t(name):
    h = hdr[name]
    return data[base + h["data_offsets"][0]: base + h["data_offsets"][1]], h["shape"]
tensors, new_prompts = [], []
for k, (pi, m) in enumerate(CUTS):
    ids, _ = t(f"ids_{pi}")
    ti, sh = t(f"topk_ids_{pi}")
    tl, _ = t(f"topk_logprobs_{pi}")
    K = sh[1]
    tensors += [(f"ids_{k}", "I32", [m], ids[: 4 * m]), (f"topk_ids_{k}", "I32", [m, K], ti[: 4 * m * K]),
                (f"topk_logprobs_{k}", "F32", [m, K], tl[: 4 * m * K])]
    p = {kk: v for kk, v in prompts[pi].items() if kk not in ("text", "text_truncated_from_chars", "reuse_split")}
    p.update(name=f"{prompts[pi]['name']}_first{m}", n_tokens=m, n_prompt=m, continuation=0)
    new_prompts.append(p)
meta["prompts"] = json.dumps(new_prompts)
meta["note"] = "prefix-truncated from " + src.split("/")[-1]
out, off = {"__metadata__": meta}, 0
for name, dt, sh, b in tensors:
    out[name] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + len(b)]}
    off += len(b)
h = json.dumps(out).encode()
h += b" " * (-len(h) % 8)
with open(dst, "wb") as o:
    o.write(struct.pack("<Q", len(h)) + h)
    for *_, b in tensors:
        o.write(b)
print("wrote", dst, [m for _, m in CUTS])
