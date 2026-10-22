#!/usr/bin/env python3
"""Make a "1 % partial-scale" mutant of a model directory without touching the original weights or the engine.

The mutant is the tp=1 equivalent of `INTARSIA_TP_TEST_BUG=scale:F@1`: on every odd-numbered layer, the per-tensor
scale of each row-parallel output projection (the projections whose partial sums the TP all-reduce adds: fp8
`self_attn.o_proj` / `linear_attn.out_proj` `weight_scale`, NVFP4 `mlp.experts.*.down_proj` and
`mlp.shared_expert.down_proj` `weight_scale_2`) is multiplied by F. Half the layers' residual updates come out F times
too large, the way half the TP partials do under the TP hook.

The output directory holds symlinks to every file of the source directory, a copied `model.safetensors.index.json` whose
weight map points the mutated scales at `zz-scale-mutant.safetensors`, and that file. intarsia's loader opens every file
named in the weight map, sorted by name, and a later file's tensor replaces an earlier one's of the same name
(`crates/intarsia-models/src/safetensors.rs`), so the `zz-` file wins. Run `--check` afterwards: it re-reads the overlay
and asserts every mutated value is F times the original.

Usage: make_scale_mutant.py SRC_MODEL_DIR OUT_DIR FACTOR [--layers odd|all]
"""
import json
import os
import re
import struct
import sys


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_scalar_f32(path, hdr, base, name):
    e = hdr[name]
    assert e["dtype"] == "F32" and e["shape"] == [], f"{name}: {e['dtype']} {e['shape']} (want an F32 scalar)"
    b, _ = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + b)
        return struct.unpack("<f", f.read(4))[0]


PAT = re.compile(
    r"layers\.(\d+)\.(self_attn\.o_proj\.weight_scale|linear_attn\.out_proj\.weight_scale|"
    r"mlp\.experts\.\d+\.down_proj\.weight_scale_2|mlp\.shared_expert\.down_proj\.weight_scale_2)$"
)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    layers = "all" if "--layers=all" in sys.argv or ("--layers" in sys.argv and "all" in sys.argv) else "odd"
    if len(args) < 3:
        sys.exit(__doc__)
    src, out, factor = os.path.abspath(args[0]), os.path.abspath(args[1]), float(args[2])
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    targets = []
    for k in sorted(wm):
        m = PAT.search(k)
        if m and (layers == "all" or int(m.group(1)) % 2 == 1):
            targets.append(k)
    if not targets:
        sys.exit("no row-parallel output scales found")
    hdrs = {}
    vals = {}
    for k in targets:
        f = wm[k]
        if f not in hdrs:
            hdrs[f] = read_header(os.path.join(src, f))
        h, base = hdrs[f]
        vals[k] = read_scalar_f32(os.path.join(src, f), h, base, k)
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(src):
        if name in ("model.safetensors.index.json",) or name.startswith("."):
            continue
        dst = os.path.join(out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src, name), dst)
    # the mutant shard: F32 scalars, header padded to 8 bytes
    header, off, blob = {}, 0, b""
    for k in targets:
        header[k] = {"dtype": "F32", "shape": [], "data_offsets": [off, off + 4]}
        blob += struct.pack("<f", vals[k] * factor)
        off += 4
    header["__metadata__"] = {"format": "pt", "fx1_scale_mutant": f"factor={factor} layers={layers} tensors={len(targets)}"}
    hj = json.dumps(header).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    mf = "zz-scale-mutant.safetensors"
    with open(os.path.join(out, mf), "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(blob)
    for k in targets:
        wm[k] = mf
    idx.setdefault("metadata", {})["fx1_scale_mutant"] = header["__metadata__"]["fx1_scale_mutant"]
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f, indent=1)
    # check: re-read the overlay the way the loader resolves names (sorted files, last wins)
    h, base = read_header(os.path.join(out, mf))
    bad = [k for k in targets if read_scalar_f32(os.path.join(out, mf), h, base, k) != struct.unpack("<f", struct.pack("<f", vals[k] * factor))[0]]
    assert not bad, f"mutant values not written: {bad[:3]}"
    files = sorted(set(wm.values()))
    assert files[-1] == mf, f"{mf} does not sort last among {files}"
    print(f"{out}: {len(targets)} scales x{factor} ({layers} layers); e.g. {targets[0]} {vals[targets[0]]:.6g} -> {vals[targets[0]] * factor:.6g}")


if __name__ == "__main__":
    main()
