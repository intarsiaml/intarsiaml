#!/usr/bin/env python3
"""Copy named tensors out of a (sharded) safetensors checkpoint into one small safetensors file, raw bytes, stdlib only.

Used for the weight-gated layer parity test (crates/intarsia-backend-cpu/tests/dsv41_weights.rs): the Mia EXL3 checkpoint
is 100+ GB, layers 0-2 without the routed experts are a few hundred MB. Run next to the checkpoint (spark1), copy the
result to the machine that runs the test.

  extract_weights.py CKPT_DIR OUT.safetensors PATTERN [PATTERN ...]    PATTERN: exact name, prefix ending in '.', or
                                                                       'regex:<re>'; prefix with '!' to exclude.
                                                                       'rows:NAME:i,j,k' copies those rows of a 2-D+ tensor
                                                                       as NAME (shape [len, ...], in the order given).
  extract_weights.py CKPT_DIR - --list PATTERN ...                      list matches and total bytes only
"""
import json, re, struct, sys
from pathlib import Path


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def main():
    ck, out, *pats = sys.argv[1:]
    listing = "--list" in pats
    pats = [p for p in pats if p != "--list"]
    ck = Path(ck)
    wm = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]

    def hit(name, pat):
        if pat.startswith("regex:"):
            return re.search(pat[6:], name) is not None
        return name == pat or (pat.endswith(".") and name.startswith(pat))

    rowsel = {}
    for p in pats:
        if p.startswith("rows:"):
            _, nm, ids = p.split(":")
            rowsel[nm] = [int(i) for i in ids.split(",")]
    pats = [p for p in pats if not p.startswith("rows:")] + list(rowsel)
    names = [k for k in wm if any(hit(k, p) for p in pats if not p.startswith("!")) and not any(hit(k, p[1:]) for p in pats if p.startswith("!"))]
    names.sort()
    heads, tensors, total = {}, {}, 0
    for k in names:
        fn = wm[k]
        if fn not in heads:
            heads[fn] = header(ck / fn)
        n, h = heads[fn]
        v = h[k]
        s, e = v["data_offsets"]
        if k in rowsel:
            rb = (e - s) // v["shape"][0]
            v = dict(v, shape=[len(rowsel[k])] + v["shape"][1:])
            tensors[k] = (fn, 8 + n + s, rb * len(rowsel[k]), v, rb)
            total += rb * len(rowsel[k])
            continue
        tensors[k] = (fn, 8 + n + s, e - s, v)
        total += e - s
    print(f"{len(names)} tensors, {total/1e6:.1f} MB", file=sys.stderr)
    if listing:
        for k in names:
            print(k, tensors[k][3]["dtype"], tensors[k][3]["shape"], tensors[k][2])
        return
    hdr, off = {"__metadata__": {"source": str(ck), "tool": "tools/dsv41/extract_weights.py"}}, 0
    for k in names:
        v = tensors[k][3]
        hdr[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + tensors[k][2]]}
        off += tensors[k][2]
    hb = json.dumps(hdr).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out, "wb") as o:
        o.write(struct.pack("<Q", len(hb)) + hb)
        for k in names:
            fn, pos, ln, *rest = tensors[k]
            with open(ck / fn, "rb") as f:
                if k in rowsel:
                    for i in rowsel[k]:
                        f.seek(pos + i * rest[1])
                        o.write(f.read(rest[1]))
                else:
                    f.seek(pos)
                    o.write(f.read(ln))


main()
