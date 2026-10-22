#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Lane ck: add one synthetic long prompt to an intarsia-golden-topk-v1 file so `accuracy --mode prefill --write-self-golden`
prefills it in several chunks at every chunk size under test (the 4k goldens fit one 4096/8192 chunk).

    make_synth.py IN.safetensors OUT.safetensors [N_TOKENS=16000]

The new prompt "synth_<N>" (kind raw) is the token ids of the file's prompts 4k docs / code / chat concatenated and cycled to N
tokens. Its top-k tensors are placeholders (ids 0, log-prob -3): only the engine's own dump is used, the golden's KL for that
prompt is meaningless. Needs the file to hold the three 4k prompts long_docs_4k, long_code_4k, chat_reuse_4k.
"""
import json, struct, sys
import numpy as np

DT = {"I32": np.int32, "F32": np.float32}


def read(path):
    b = open(path, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8:8 + n])
    return h, b[8 + n:]


def main():
    src, out = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
    h, data = read(src)
    meta = h.pop("__metadata__")
    prompts = json.loads(meta["prompts"])
    ids = {}
    for name in ("long_docs_4k", "long_code_4k", "chat_reuse_4k"):
        j = next(i for i, p in enumerate(prompts) if p["name"] == name)
        s, e = h[f"ids_{j}"]["data_offsets"]
        ids[name] = np.frombuffer(data[s:e], dtype=np.int32)
    cat = np.concatenate([ids["long_docs_4k"], ids["long_code_4k"], ids["chat_reuse_4k"]])
    syn = np.tile(cat, N // len(cat) + 1)[:N].astype(np.int32)
    k = int(meta.get("topk", 20))
    k_id = len(prompts)
    prompts.append({"name": f"synth_{N // 1000}k", "kind": "raw", "n_tokens": N, "n_prompt": N, "continuation": 0, "text": ""})
    meta["prompts"] = json.dumps(prompts)
    meta["synth_note"] = "synth_*: ids tiled from the 4k prompts; top-k tensors are placeholders (lane ck)"
    tensors = [(k_, v, data[v["data_offsets"][0]:v["data_offsets"][1]]) for k_, v in h.items()]
    add = [(f"ids_{k_id}", "I32", [N], syn.tobytes()),
           (f"topk_ids_{k_id}", "I32", [N, k], np.zeros((N, k), np.int32).tobytes()),
           (f"topk_logprobs_{k_id}", "F32", [N, k], np.full((N, k), -3.0, np.float32).tobytes())]
    header, off, blob = {"__metadata__": meta}, 0, []
    for name, v, raw in tensors:
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
    print(f"{out}: {len(prompts)} prompts, added synth_{N // 1000}k ({N} tokens)")


main()
