"""Speed summary of lane ink-hip-overlap's paired windows (S1 / S2 / S3; from ink-hip-bpf's speed_sum.py): per window and arm
(A = main5, B = o4: overlap on by default) the real-prompt levels (aggregate, TTFT median / max, ITL p50 / p99), the 1k
filler sweep (localeval: aggregate decode tok/s, TTFT median), the pure-prefill rows (prompt tok/s) and the VRAM peak; then the
paired ratios X / A per window and their mean with a 95 % t interval over the windows.
Adds the real-prompt TTFT p95 (from the replies), ITL p90, and the gate-4 verdict lines.
  python3 speed_sum.py s1 s2 s3"""
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
    for arm in ("A", "B"):
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
                rj = f[0].replace(".meta.json", ".json")
                if os.path.exists(rj):
                    recs = [r for r in json.load(open(rj)) if isinstance(r, dict)]
                    tt = sorted(r["ttft"] for r in recs)
                    g = sorted(x for r in recs for x in r.get("gaps", []))
                    if tt:
                        m[f"real{n}.ttft95"] = tt[min(len(tt) - 1, int(math.ceil(0.95 * len(tt))) - 1)]
                    if g:
                        m[f"real{n}.itl90"] = g[int(0.9 * len(g))]
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
            row = [f"{data[w].get(a, {}).get(k)}" for a in ("A", "B")]
            print(f"  {k:<16} A {row[0]:>8}  B {row[1]:>8}")
    print("== paired X / A (mean over windows, 95 % t interval); for TTFT / ITL lower is better")
    verdict = []
    for x in ("B",):
        for k in keys:
            rs = [data[w][x][k] / data[w]["A"][k] for w in wins
                  if x in data[w] and "A" in data[w] and data[w][x].get(k) and data[w]["A"].get(k)]
            if not rs:
                continue
            mu, lo, hi = ci(rs)
            iv = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "(1 rep)"
            print(f"  {x}/A {k:<16} {mu:.3f} {iv}  n={len(rs)}  per window {', '.join(f'{r:.3f}' for r in rs)}")
            if lo is None:
                continue
            if k in ("real8.itl99", "real16.itl99"):
                verdict.append((k, "ITL p99 upper <= 0.70", hi <= 0.70, f"{hi:.3f}"))
            elif k.endswith(".agg"):
                verdict.append((k, "aggregate upper >= 0.99", hi >= 0.99, f"{hi:.3f}"))
            elif k.endswith(".ttft50"):
                verdict.append((k, "TTFT p50 lower <= 1.01", lo <= 1.01, f"{lo:.3f}"))
    print("== gate 4 (rule of section 4)")
    for k, what, ok, v in verdict:
        print(f"  {'PASS' if ok else 'FAIL'}  {k:<14} {what}: {v}")


if __name__ == "__main__":
    main()
