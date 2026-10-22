#!/usr/bin/env python3
"""fx5c: build a committed `continue` oracle file (format intarsia-continue-oracle-v1) from a frozen sequence and the CPU oracle's dump.

usage: make_continue_oracle.py SEQ.json ORACLE_DUMP.safetensors LONG_GOLDEN.safetensors PROMPT OUT_DIR [--machine TEXT] [--run TEXT]

SEQ.json: written by `accuracy --reuse-dump` (turn2, stream, split, row0, m). ORACLE_DUMP: `accuracy --backend cpu --mode prefill
--write-self-golden` over `make_oracle_input.py`'s one-prompt golden of the same sequence. The file keeps only what the gate needs:
the frozen reply (turn2[split:split+16]) and stream tokens, the model, the prompt name and split, a fingerprint of the golden prompt's
token ids, and the oracle's top-64 rows for the stream positions (32 x 64, about 16 KB). Writes OUT_DIR/<stem>.continue-oracle.safetensors
and a sidecar .json with its sha256 and provenance.
"""
import hashlib
import json
import struct
import sys

import numpy as np


def read(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        data = f.read()

    def t(name):
        e = h[name]
        b, en = e["data_offsets"]
        return np.frombuffer(data[b:en], dtype={"I32": np.int32, "F32": np.float32}[e["dtype"]]).reshape(e["shape"])

    return h["__metadata__"], t


def fingerprint(ids):  # FNV-1a over the little-endian u32 bytes: the same as accuracy.rs `ids_fingerprint`
    h = 0xCBF29CE484222325
    for t in ids:
        for b in int(t).to_bytes(4, "little"):
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def main():
    a = sys.argv[1:]
    opts = {}
    while "--machine" in a or "--run" in a:
        for o in ("--machine", "--run"):
            if o in a:
                i = a.index(o)
                opts[o] = a[i + 1]
                del a[i : i + 2]
    if len(a) != 5:
        sys.exit(__doc__)
    seq_p, ora_p, gold_p, prompt, out_dir = a
    seq = json.load(open(seq_p))
    om, ot = read(ora_p)
    ids, oid, olp = ot("ids_0"), ot("topk_ids_0"), ot("topk_logprobs_0")
    assert list(ids) == seq["turn2"] + seq["stream"], "oracle sequence != seq.json"
    gm, gt = read(gold_p)
    names = [p["name"] for p in json.loads(gm["prompts"])]
    pid = gt(f"ids_{names.index(prompt)}")
    split, m, row0 = seq["split"], seq["m"], seq["row0"]
    reply = seq["turn2"][split : split + len(seq["fed"]) + 1]
    assert seq["turn2"][:split] + seq["turn2"][split + len(reply) :] == [int(x) for x in pid], "turn2 is not the golden prompt with the reply inserted"
    assert seq["turn2"][split : split + len(seq["fed"])] == seq["fed"]
    rows_ids, rows_lp = oid[row0 : row0 + m].astype(np.int32), olp[row0 : row0 + m].astype(np.float32)
    meta = {
        "format": "intarsia-continue-oracle-v1",
        "model": gm["model"],
        "prompt": prompt,
        "split": str(split),
        "reply": json.dumps(reply),
        "stream": json.dumps(seq["stream"]),
        "prompt_ids_fingerprint": fingerprint(pid),
        "prefill_chunk_when_frozen": str(seq["chunk"]),
        "row0": str(row0),
        "oracle": "intarsia accuracy --backend cpu (f64 CPU oracle, fx1 recipe: prefill chunk 256, top-64), one prefill over turn2 + stream",
        "oracle_machine": opts.get("--machine", "unrecorded"),
        "oracle_code": om.get("code_stamp", om.get("code", "see oracle log")),
        "run": opts.get("--run", ""),
        "note": "row j = distribution stream[j] is sampled from, position row0 + j; rows are the oracle's top-64 log-probs (ids -1 padded), tail = 1 - sum",
    }
    tensors = [("topk_ids", "I32", list(rows_ids.shape), rows_ids.tobytes()), ("topk_logprobs", "F32", list(rows_lp.shape), rows_lp.tobytes())]
    header, off, blob = {"__metadata__": meta}, 0, []
    for name, dt, shape, raw in tensors:
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blob.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    hj += b" " * ((8 - (8 + len(hj)) % 8) % 8)
    stem = f"{gm['model'].split('/')[-1].lower().replace('qwen3.6', 'qwen36')}-continue-{prompt}-s{split}"
    path = f"{out_dir}/{stem}.continue-oracle.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for r in blob:
            f.write(r)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    side = {"file": path.split("/")[-1], "sha256": sha, "bytes": len(open(path, "rb").read()), "built_by": "scripts/fx5c/make_continue_oracle.py", "inputs": {"seq": seq_p.split("/")[-1], "oracle_dump": ora_p.split("/")[-1], "golden": gold_p.split("/")[-1]}, **{k: v for k, v in meta.items() if k not in ("reply", "stream", "note")}}
    json.dump(side, open(path + ".json", "w"), indent=2)
    print(path, sha, side["bytes"], "bytes")


if __name__ == "__main__":
    main()
