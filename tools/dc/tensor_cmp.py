"""Lane dc: compare named tensors of two safetensors checkpoints byte for byte (stdlib only; reads headers + ranges).
usage: tensor_cmp.py <a_dir_or_file> <b_dir_or_file> <name_a>=<name_b> ...   (a dir with model.safetensors.index.json is resolved)"""
import json, os, struct, sys

def open_ckpt(p):
    if os.path.isdir(p):
        idx = os.path.join(p, "model.safetensors.index.json")
        files = [os.path.join(p, f) for f in sorted(set(json.load(open(idx))["weight_map"].values()))] if os.path.exists(idx) else [os.path.join(p, "model.safetensors")]
    else:
        files = [p]
    table = {}
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                table[k] = (f, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v["dtype"], v["shape"])
    return table

def read(t, lo=0, hi=None):
    f, off, ln, _, _ = t
    with open(f, "rb") as fh:
        fh.seek(off + lo); return fh.read((ln if hi is None else hi) - lo)

a, b = open_ckpt(sys.argv[1]), open_ckpt(sys.argv[2])
for pair in sys.argv[3:]:
    na, nb = pair.split("=")
    ta, tb = a[na], b[nb]
    print(f"{na} {ta[3]}{ta[4]} vs {nb} {tb[3]}{tb[4]}", end="  ")
    if ta[2] != tb[2]:
        print("SIZE DIFFERS"); continue
    row = ta[2] // ta[4][0] if len(ta[4]) == 2 else ta[2]
    diff_rows, step = [], 1 << 26
    for lo in range(0, ta[2], step):
        x, y = read(ta, lo, min(lo + step, ta[2])), read(tb, lo, min(lo + step, tb[2]))
        if x != y:
            for r in range(len(x) // row):
                if x[r * row:(r + 1) * row] != y[r * row:(r + 1) * row]:
                    diff_rows.append(lo // row + r)
    print("IDENTICAL" if not diff_rows else f"DIFFERENT rows: {len(diff_rows)} first {diff_rows[:8]} mask(151669) differs: {151669 in diff_rows}")
