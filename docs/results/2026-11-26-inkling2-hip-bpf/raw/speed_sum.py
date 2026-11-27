"""Speed summary of lane ink-hip-bpf's paired windows (W6 / W7 / W8): per window and arm (A = main3, B = branch packing at 2048
rows, R = branch packing with the 4096-row workspace) the real-prompt levels (aggregate, TTFT median / max, ITL p50 / p99), the 1k
filler sweep (localeval: aggregate decode tok/s, TTFT median), the pure-prefill rows (prompt tok/s) and the VRAM peak; then the
paired ratios X / A per window and their mean with a 95 % t interval over the windows.
  python3 speed_sum.py w6 w7 w8"""
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def load(win):
    d = os.path.join(HERE, win)
    out = {}
    for arm in ("A", "B", "R"):
        m = {}
        for n in (4, 8, 16):
            f = glob.glob(os.path.join(d, f"{arm}-real*-r1-n{n}.meta.json"))
            if f:
                j = json.load(open(f[0]))
                m[f"real{n}.agg"] = j["aggregate_tok_s"]
                m[f"real{n}.ttft50"] = j.get("ttft_s_median")
                m[f"real{n}.ttftmax"] = j.get("ttft_s_max")
                m[f"real{n}.itl50"] = j.get("itl_s_p50") or j.get("itl_p50")
                m[f"real{n}.itl99"] = j.get("itl_s_p99") or j.get("itl_p99")
        f = os.path.join(d, f"{arm}-fill-r1.result.json")
        if os.path.exists(f):
            for c in json.load(open(f))["speed"]["concurrency"]:
                s = c["streams"]
                m[f"fill{s}.agg"] = c["aggregate_decode_tok_s"]
                m[f"fill{s}.ttft50"] = c["ttft_s"]["median"]
                m[f"fill{s}.ttftmax"] = c["ttft_s"]["max"]
        for f in glob.glob(os.path.join(d, f"{arm}-bitwise-pp_*.meta.json")):
            k, L = re.search(r"pp_(\d+)_(\d+)", f).groups()
            m[f"pp{k}x{L}"] = json.load(open(f))["prompt_tok_s"]
        f = os.path.join(d, f"{arm}-stop.out")
        if os.path.exists(f):
            v = re.search(r"card0 (\d+) MiB", open(f).read())
            if v:
                m["vram"] = int(v.group(1))
        if m:
            out[arm] = m
    return out


def ci(xs):
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, None, None
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    h = T95[n - 1] * sd / math.sqrt(n)
    return mu, mu - h, mu + h


def main():
    wins = sys.argv[1:]
    data = {w: load(w) for w in wins}
    keys = sorted({k for w in data.values() for a in w.values() for k in a}, key=lambda k: (re.sub(r"\d+", "", k), [int(x) for x in re.findall(r"\d+", k)]))
    for w in wins:
        print(f"== {w}: arms {sorted(data[w])}")
        for k in keys:
            row = [f"{data[w].get(a, {}).get(k)}" for a in ("A", "B", "R")]
            print(f"  {k:<16} A {row[0]:>8}  B {row[1]:>8}  R {row[2]:>8}")
    print("== paired X / A (mean over windows, 95 % t interval); for TTFT / ITL lower is better")
    for x in ("B", "R"):
        for k in keys:
            rs = [data[w][x][k] / data[w]["A"][k] for w in wins
                  if x in data[w] and "A" in data[w] and data[w][x].get(k) and data[w]["A"].get(k)]
            if not rs:
                continue
            mu, lo, hi = ci(rs)
            iv = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "(1 rep)"
            print(f"  {x}/A {k:<16} {mu:.3f} {iv}  n={len(rs)}  per window {', '.join(f'{r:.3f}' for r in rs)}")


if __name__ == "__main__":
    main()
