#!/usr/bin/env python3
"""fx5c: one-prompt intarsia-golden-topk-v1 file holding the teacher-forced `continue` sequence, as input for the CPU oracle
(`intarsia accuracy --backend cpu --mode prefill --write-self-golden ...`).

usage: make_oracle_input.py SEQ.json LONG_GOLDEN.safetensors OUT.safetensors

The sequence is turn2 + stream (all m stream tokens, so that row m-1 exists too). The reference arrays are filler (the
oracle's own rows are the result, the tool's verdict on the filler is meaningless); the metadata is the long golden's.
"""
import json
import struct
import sys

import numpy as np


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    seq = json.load(open(sys.argv[1]))
    ids = np.array(seq["turn2"] + seq["stream"], dtype=np.int32)
    n = len(ids)
    with open(sys.argv[2], "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        meta = dict(json.loads(f.read(hl))["__metadata__"])
    meta["prompts"] = json.dumps([{"name": "fx5c_continue", "kind": "raw", "n_tokens": n, "n_prompt": n, "continuation": 0, "text": ""}])
    meta["note"] = "fx5c: filler reference arrays; teacher-forced continue sequence for the CPU oracle"
    k = 20
    tensors = [
        ("ids_0", "I32", [n], ids.tobytes()),
        ("topk_ids_0", "I32", [n, k], np.tile(np.arange(k, dtype=np.int32), (n, 1)).tobytes()),
        ("topk_logprobs_0", "F32", [n, k], np.full((n, k), -5.0, dtype=np.float32).tobytes()),
    ]
    header, off, blob = {"__metadata__": meta}, 0, []
    for name, dt, shape, raw in tensors:
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blob.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    hj += b" " * ((8 - (8 + len(hj)) % 8) % 8)
    with open(sys.argv[3], "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for r in blob:
            f.write(r)
    print(f"{sys.argv[3]}: 1 prompt, {n} tokens (turn2 {len(seq['turn2'])} + stream {len(seq['stream'])}), rows of interest {seq['row0']}..{seq['row0'] + seq['m'] - 1}")


if __name__ == "__main__":
    main()
