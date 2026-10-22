#!/usr/bin/env python3
"""fx6: aggregate Metal sameslot runs (scripts/metal-speed-gate/sameslot.sh output dirs) for the standings.

usage: aggregate-metal.py <dir with run1/ run2/ ...> [--out table.md]

Per visit (one server start, 3 reps): raw decode median per size, the refbw.py machine-state factor (median reference GB/s of that
arm's bracketing runs divided by scripts/metal-speed-gate/ref-baseline.json) and the clock-normalised decode (raw / factor), applied
to EVERY engine, not only intarsia. Prefill is reported raw (and marginal), unnormalised: the reference is a memory-bound GEMV, so
dividing a compute-bound prefill by it would be unfounded. Cells: median over visits with the min-max band; n = visits.
Ranks are by median, tie when the bands overlap.
"""
import glob, gzip, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = json.load(open(os.path.join(HERE, "..", "metal-speed-gate", "ref-baseline.json")))["gbs"]
SIZES = [128, 1024, 4096, 16384]


def engine(label):
    for p, e in (("intarsia", "intarsia"), ("mlx-lm", "mlx-lm"), ("llamacpp", "llama.cpp")):
        if label.startswith(p):
            return e
    return None


def ref(path):
    v = []
    if os.path.exists(path):
        for line in open(path):
            try:
                v.append(float(json.loads(line)["gbs"]))
            except Exception:
                pass
    return st.median(v) if v else None


def macmon(path):
    for p in (path, path + ".gz"):
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            busy = []
            for line in op(p, "rt"):
                try:
                    d = json.loads(line)
                    mhz, util = d["gpu_usage"]
                    if util >= 0.5:
                        busy.append(mhz)
                except Exception:
                    pass
            return st.median(busy) if busy else None
    return None


def band(vals, nd=1):
    if not vals:
        return "-"
    return f"{st.median(vals):.{nd}f} ({min(vals):.{nd}f}-{max(vals):.{nd}f})"


def main():
    d = sys.argv[1]
    visits = []  # (engine, label, run, factor, mhz, rows)
    for f in sorted(glob.glob(os.path.join(d, "run*", "*-localeval.json"))):
        run = os.path.basename(os.path.dirname(f))
        label = os.path.basename(f)[: -len("-localeval.json")]
        e = engine(label)
        j = json.load(open(f))
        rows = {r["target_prompt_tokens"]: r for r in j["speed"]["rows"]}
        base = os.path.join(os.path.dirname(f), label)
        r = ref(base + "-ref.jsonl")
        visits.append((e, f"{run}/{label}", run, r / BASE if r else None, macmon(base + "-macmon.jsonl"), r, rows))
    out = []
    out.append(f"Reference baseline {BASE} GB/s (ref-baseline.json). State factor = arm reference / baseline; valid 0.70-1.30.\n")
    out.append("**Per visit**\n")
    out.append("| visit | ref GB/s | factor | busy GPU MHz | " + " | ".join(f"decode {s} raw / norm" for s in SIZES) + " |")
    out.append("|---|---:|---:|---:|" + "---:|" * len(SIZES))
    for e, lab, run, fac, mhz, r, rows in visits:
        cells = []
        for s in SIZES:
            x = rows.get(s)
            if not x:
                cells.append("-")
                continue
            raw = x["decode_tok_s"]["median"]
            cells.append(f"{raw:.1f} / {raw / fac:.1f}" if fac else f"{raw:.1f} / -")
        head = f"| {lab} | {r:.1f} | {fac:.3f} | {mhz:.0f} | " if (r and mhz) else f"| {lab} | - | - | {mhz or 0:.0f} | "
        out.append(head + " | ".join(cells) + " |")
    engines = [e for e in ("intarsia", "mlx-lm", "llama.cpp") if any(v[0] == e for v in visits)]

    def collect(e, s, kind):
        vals = []
        for ee, lab, run, fac, mhz, r, rows in visits:
            if ee != e or s not in rows:
                continue
            x = rows[s]
            if kind == "raw":
                vals.append(x["decode_tok_s"]["median"])
            elif kind == "norm" and fac and 0.70 <= fac <= 1.30:
                vals.append(x["decode_tok_s"]["median"] / fac)
            elif kind == "prefill":
                vals.append(x["prefill_tok_s"]["median"])
            elif kind == "marginal" and x.get("marginal_prefill_tok_s"):
                vals.append(x["marginal_prefill_tok_s"])
        return vals

    ranks = {}
    for title, kind, nd, sizes in (("Decode tok/s, raw", "raw", 1, SIZES), ("Decode tok/s, clock-normalised (raw / state factor, every engine)", "norm", 1, SIZES),
                                   ("Prefill tok/s, raw (per request)", "prefill", 0, SIZES[1:]), ("Marginal prefill tok/s, raw", "marginal", 0, SIZES[1:])):
        out.append(f"\n**{title}** - median over visits (min-max), n visits per cell\n")
        out.append("| engine | n | " + " | ".join(str(s) for s in sizes) + " |")
        out.append("|---|---:|" + "---:|" * len(sizes))
        for e in engines:
            n = len(collect(e, sizes[0], kind))
            out.append(f"| {e} | {n} | " + " | ".join(band(collect(e, s, kind), nd) for s in sizes) + " |")
        # rank line
        rk = []
        for s in sizes:
            ms = {e: collect(e, s, kind) for e in engines if collect(e, s, kind)}
            order = sorted(ms, key=lambda e: -st.median(ms[e]))
            parts = []
            for i, e in enumerate(order):
                if i == 0:
                    parts.append(e)
                else:
                    tie = not (min(ms[order[i - 1]]) > max(ms[e]))
                    parts.append((" ~ " if tie else " > ") + e)
            rk.append(f"{s}: " + "".join(parts))
        out.append("\nOrder (~ = bands overlap, a tie): " + "; ".join(rk))
        ranks[kind] = rk
    txt = "\n".join(out)
    print(txt)
    if "--out" in sys.argv:
        open(sys.argv[sys.argv.index("--out") + 1], "w").write(txt + "\n")


if __name__ == "__main__":
    main()
