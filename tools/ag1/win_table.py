#!/usr/bin/env python3
"""Lane ag1: markdown tables of one window of scripts/ag1/cat-window.sh (a successor of tools/au1/win_table.py): decode tok/s per category and arm
(pooled per round as tokens / sum of decode times, the mean of the rounds a category has), auto vs MTP k2 and vs the faster of MTP k2 / DFlash2 k7, the
pre-registered rule's verdict per category, the same per round, the share of DFlash passes of the auto arm (its per-pass logs), and the mechanism
check: auto pinned to the MTP head (`autotrmtp`) against `mtp2`, pooled over the 7 categories (sum of tokens / sum of decode seconds), against the
mean of mtp2's rounds (autotrmtp has one round), plus its per-category ratios.
usage: win_table.py <raw/W> [auto arm name, default auto]"""
import glob, json, os, re, sys

raw = sys.argv[1].rstrip("/")
auto = sys.argv[2] if len(sys.argv) > 2 else "auto"
res = "/opt/code/deepeval/results"
GEN = 512
CATS = [("ho-chat", 4), ("ho-code", 2), ("ho-prose", 4), ("ho-recall", 2), ("rcp-code", 8), ("rcp-prose", 8), ("rcp-recall", 8)]

steps = open(os.path.join(raw, "steps.log")).read()
runs = re.findall(r"round (\d+) arm (\S+) (\S+): localeval exit (\d+) (results/\S+)", steps)
# (arm, cat, round) -> (tokens, seconds, problems)
cell = {}
problems = []
for r, arm, cat, rc, path in runs:
    cases = json.load(open(os.path.join(os.path.dirname(res), path)))["datasets"][0]["cases"]
    toks = secs = 0.0
    for c in cases:
        n = c["speed"]["completion_tokens"]
        if n != GEN:
            problems.append(f"{arm} {cat} r{r} {c['name']}: {n} tokens")
        if c["speed"]["decode_tok_s"]:
            toks += n - 1
            secs += (n - 1) / c["speed"]["decode_tok_s"]
    cell[(arm, cat, int(r))] = (toks, secs, sorted(c["name"] for c in cases))
arms = sorted({a for a, _, _ in cell})
rounds = sorted({r for _, _, r in cell})
for c, _ in CATS:
    ref = cell.get(("mtp2", c, rounds[0]), (0, 0, None))[2]
    for (a, cc, r), v in cell.items():
        if cc == c and v[2] != ref:
            problems.append(f"{a} {c} r{r}: rows differ from mtp2 r{rounds[0]}")
print(f"window {os.path.basename(raw)}: {len(runs)} localeval runs, arms {arms}, rounds {rounds}; problems: {len(problems)}", file=sys.stderr)
for p in problems[:20]:
    print("PROBLEM", p, file=sys.stderr)


def rate(arm, cat, rs):
    """tok/s of an arm in a category: the mean of its rounds' pooled rates (rs: the rounds to take)."""
    v = [cell[(arm, cat, r)][0] / cell[(arm, cat, r)][1] for r in rs if (arm, cat, r) in cell]
    return sum(v) / len(v) if v else None


# DFlash passes of the auto arm, by category (its pass logs: one "spec pass" line per pass, one "spec auto:" line per request, in request order)
D = {c: [0, 0] for c, _ in CATS}
for r in rounds:
    f = f"{raw}/pass-{auto}-r{r}.log"
    if not os.path.exists(f):
        continue
    reqs, cur = [], []
    for ln in open(f, errors="replace"):
        if ln.startswith("spec pass "):
            cur.append(ln.split()[2])
        elif ln.startswith("spec auto:"):
            reqs.append(cur)
            cur = []
    i = 0
    for c, n in CATS:
        for q in reqs[i:i + n]:
            D[c][0] += q.count("D")
            D[c][1] += len(q)
        i += n


def verdict(rs, label):
    print(f"\n{label}\n")
    print("| category | mtp2 | dfl7 | auto | auto / mtp2 | auto / best | rule | DFlash passes of auto |")
    print("|---|---:|---:|---:|---:|---:|---|---:|")
    allok = True
    for c, _ in CATS:
        m, f, a = rate("mtp2", c, rs), rate("dfl7", c, rs), rate(auto, c, rs)
        if None in (m, f, a):
            print(f"| {c} | missing | | | | | n/a | |")
            allok = False
            continue
        best = max(m, f)
        r1, r2 = a / m, a / best
        ok = r1 >= 0.99 and r2 >= 0.97
        allok &= ok
        share = f"{100 * D[c][0] / max(D[c][1], 1):.0f}%" if D[c][1] else "-"
        print(f"| {c} | {m:.1f} | {f:.1f} | **{a:.1f}** | {r1:.4f} | {r2:.4f} | {'pass' if ok else 'FAIL'} | {share} |")
    print(f"\nrule on {label}: **{'MET' if allok else 'NOT MET'}**")
    return allok


ok_all = verdict(rounds, f"rounds {rounds} together (the mean of each arm's rounds: the decision)")
if len(rounds) > 1:
    for r in rounds:
        verdict([r], f"round {r} alone")

# mechanism check: auto pinned to the MTP head against mtp2, pooled over the 7 categories
pin = "autotrmtp"
if any(a == pin for a, _, _ in cell):
    pin_rounds = sorted({r for a, _, r in cell if a == pin})
    toks = sum(cell[(pin, c, r)][0] for c, _ in CATS for r in pin_rounds if (pin, c, r) in cell)
    secs = sum(cell[(pin, c, r)][1] for c, _ in CATS for r in pin_rounds if (pin, c, r) in cell)
    mt = {}
    for label, rs in (("mean of mtp2's rounds", rounds), ("mtp2 of the same round", pin_rounds)):
        t = sum(cell[("mtp2", c, r)][0] for c, _ in CATS for r in rs if ("mtp2", c, r) in cell)
        s = sum(cell[("mtp2", c, r)][1] for c, _ in CATS for r in rs if ("mtp2", c, r) in cell)
        mt[label] = (t / s) if s else None
    print(f"\nMechanism check (auto pinned to the MTP head, round(s) {pin_rounds}, pooled over the categories present):")
    print(f"\n| comparison | {pin} tok/s | mtp2 tok/s | ratio |\n|---|---:|---:|---:|")
    for label, v in mt.items():
        print(f"| vs {label} | {toks / secs:.2f} | {v:.2f} | {toks / secs / v:.4f} |")
    pooled_mean = toks / secs / mt["mean of mtp2's rounds"]
    print(f"\nmechanism check (pooled >= 0.995 x the mean of mtp2's rounds): **{'PASS' if pooled_mean >= 0.995 else 'FAIL'}** ({pooled_mean:.4f})")
    print("\n| category | " + pin + " tok/s | x mtp2 (mean of rounds) | x mtp2 (same round) |\n|---|---:|---:|---:|")
    for c, _ in CATS:
        a = rate(pin, c, pin_rounds)
        print(f"| {c} | {a:.1f} | {a / rate('mtp2', c, rounds):.4f} | {a / rate('mtp2', c, pin_rounds):.4f} |")
extra = [a for a in arms if a not in ("mtp2", "dfl7", auto, pin)]
if extra:
    print("\n| category | " + " | ".join(f"{a} tok/s (x mtp2)" for a in extra) + " |\n|---|" + "---:|" * len(extra))
    for c, _ in CATS:
        print(f"| {c} | " + " | ".join(f"{rate(a, c, rounds):.1f} ({rate(a, c, rounds) / rate('mtp2', c, rounds):.3f})" for a in extra) + " |")
sys.exit(0 if ok_all else 1)
