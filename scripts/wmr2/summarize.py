#!/usr/bin/env python3
"""Lane wmr: tables from the localeval logs of the window folders.
usage: summarize.py RAWDIR... > tables.md   (reads RAWDIR/localeval-<phase>-<tp>-<arm>-r<round>.log)
Multi-stream phases (g512*): aggregate decode tok/s per stream count. b1: single-stream TTFT, prefill tok/s, decode tok/s per prompt size.
Every cell lists the per-round values (in round order), then the mean; the delta is on the means, and the paired min/max of the per-round deltas
is given so a drift between rounds is visible."""
import re, sys, glob, os, collections, statistics as st

multi = collections.defaultdict(dict)   # (tp, streams) -> arm -> {round_key: value}
single = collections.defaultdict(dict)  # (tp, size, metric) -> arm -> {round_key: value}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "localeval-*.log"))):
        m = re.match(r"localeval-(\w+?)-(tp\d)-(\w+)-r(\d+)\.log", os.path.basename(f))
        if not m:
            continue
        phase, tp, arm, r = m.group(1), m.group(2), m.group(3), int(m.group(4))
        rk = (phase, r)
        for line in open(f, errors="replace"):
            a = re.match(r"\[localeval\] (\d+) streams: aggregate ([\d.]+) tok/s", line)
            if a:
                multi[(tp, int(a.group(1)))].setdefault(arm, {})[rk] = float(a.group(2))
            b = re.match(r"\[localeval\]\s+(\d+) tok rep \d+/\d+: prompt=(\d+) ttft=([\d.]+)s prefill=(\d+) tok/s decode=([\d.]+) tok/s", line)
            if b and phase.startswith("b1"):
                sz = int(b.group(1))
                for name, v in (("ttft_s", b.group(3)), ("prefill_tok_s", b.group(4)), ("decode_tok_s", b.group(5))):
                    single[(tp, sz, name)].setdefault(arm, {})[rk] = float(v)


def f(v):
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"


def cell(vals):
    ks = sorted(vals)
    return " / ".join(f(vals[k]) for k in ks), st.mean(vals.values())


def table(title, data, keyfmt, ref="off"):
    if not data:
        return
    print(f"\n### {title}\n")
    arms = sorted({a for v in data.values() for a in v}, key=lambda a: (a != ref, a))
    print("| config | " + " | ".join(f"{a} (rounds; mean)" for a in arms) + " | " + " | ".join(f"{a} vs {ref}" for a in arms if a != ref) + " |")
    print("|---|" + "---|" * (len(arms) + len(arms) - 1))
    for key in sorted(data):
        row, means = [], {}
        for a in arms:
            if a in data[key]:
                s, mu = cell(data[key][a]); means[a] = mu; row.append(f"{s}; **{f(mu)}**")
            else:
                row.append("-")
        deltas = []
        for a in arms:
            if a == ref:
                continue
            if a in means and ref in means:
                common = sorted(set(data[key][a]) & set(data[key][ref]))
                pd = [100 * (data[key][a][k] / data[key][ref][k] - 1) for k in common]
                rng = f" [{min(pd):+.1f}..{max(pd):+.1f}]" if len(pd) > 1 else ""
                deltas.append(f"{100 * (means[a] / means[ref] - 1):+.1f} %{rng}")
            else:
                deltas.append("-")
        print(f"| {keyfmt(key)} | " + " | ".join(row) + " | " + " | ".join(deltas) + " |")


table("Aggregate decode tok/s (1k prompt, gen 512, `--max-batch 16`, spec off)", multi, lambda k: f"{k[0]}, {k[1]} streams")
for name in ("decode_tok_s", "prefill_tok_s", "ttft_s"):
    table(f"Single stream (`--max-batch 1`), {name}", {k: v for k, v in single.items() if k[2] == name}, lambda k: f"{k[0]}, {k[1]}-token prompt")
