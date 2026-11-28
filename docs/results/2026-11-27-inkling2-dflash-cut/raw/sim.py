#!/usr/bin/env python3
"""Lane ink-dflash-cut: offline simulation of a cumulative-product cut on DFlash2 drafts (Metal, Inkling-2-Mini).

Input: a traced `generate --prompt-ids speed.json` stderr (INTARSIA_SPEC_TRACE=1 + INTARSIA_DFLASH_CONF_TRACE=1), at a cap K0 (k7 covers
every cap <= 7). Each `spec: dflash2 conf` line (per block row `tok:p_full:p_raw1:p_sel:margin[:p16]`) is paired with the next
`spec: ... token N pos P drafted D accepted A` line; prompts are split at the `decode[<name>]` lines.

Cost model per prompt (measured, k sweep #12478, raw/s2-sweep of docs/results/2026-11-26-inkling2-metal-dflash2): ms per pass = a + b * rows
(least squares over k3 / k4 / k5 / k7, rows = k + 1). A cut to n drafts: rows n + 1, tokens min(A, n) + 1 (acceptance is a prefix).
The draft itself (one pass for the block) is paid whatever n is, so it sits in `a`.

Rule: keep drafts while the running product of the signal stays >= tau (the draft that takes it below tau is dropped, DSpark's rule),
at most `cap`. Baseline: static k5 (the Metal default), tokens min(A, 5) + 1, rows 6.
Approximation: the pass population is the traced one (a cut shifts where later passes start); `check` compares the k5 traced run's own
tokens per pass with min(A7, 5) + 1 from the k7 trace.

usage: sim.py TRACE_K7.err SWEEP_DIR [TRACE_K5.err]
"""
import math, re, statistics as st, sys
from pathlib import Path

SIGNALS = {"p_full": 1, "p_raw1": 2, "p_sel": 3, "p16": 5}


def parse(path):
    prompts, cur, pend = {}, [], None
    for l in open(path, errors="replace"):
        if l.startswith("spec: dflash2 conf"):
            rows = [x.split(":") for x in l.split()[3:]]
            pend = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]) if len(r) > 5 else float("nan")) for r in rows]
            continue
        m = re.match(r"spec: \S*\s*token \d+ pos \d+ drafted (\d+) accepted (\d+) emits \d+.*? drafts \[([^\]]*)\]", l)
        if m and pend is not None:
            drafted, acc = int(m.group(1)), int(m.group(2))
            dr = [int(x) for x in m.group(3).split(",") if x.strip()]
            ok = [r[0] for r in pend[: len(dr)]] == dr
            cur.append({"conf": pend, "drafted": drafted, "acc": acc, "match": ok})
            pend = None
            continue
        m = re.match(r"decode\[(.+?)\]: (\d+) tokens, (\d+) ms", l)
        if m:
            prompts[m.group(1)] = cur
            cur = []
    return prompts


def fit(sweep):
    d = {}
    for k in (3, 4, 5, 7):
        cur = None
        for l in open(Path(sweep) / f"dfl{k}@r1.err", errors="replace"):
            m = re.match(r"decode\[(.+?)\]: (\d+) tokens, (\d+) ms", l)
            if m:
                cur = m.group(1)
                d.setdefault(cur, {})[k] = [int(m.group(3))]
            m = re.match(r"spec\[(.+?)\]: verify passes=(\d+)", l)
            if m and m.group(1) == cur:
                d[cur][k].append(int(m.group(2)))
    out = {}
    for p, v in d.items():
        if not all(k in v and len(v[k]) == 2 and v[k][1] > 0 for k in (3, 4, 5, 7)):
            continue
        xs = [k + 1 for k in (3, 4, 5, 7)]
        ys = [v[k][0] / v[k][1] for k in (3, 4, 5, 7)]
        mx, my = st.mean(xs), st.mean(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        out[p] = (my - b * mx, b)
    return out


def kept(conf, sig, tau, cap):
    prod, n = 1.0, 0
    for r in conf[:cap]:
        prod *= r[sig]
        if prod < tau:
            break
        n += 1
    return n


def speed(passes, ab, sig=None, tau=0.0, cap=5):
    a, b = ab
    toks = ms = 0.0
    for ps in passes:
        n = cap if sig is None else kept(ps["conf"], sig, tau, cap)
        toks += min(ps["acc"], n) + 1
        ms += a + b * (n + 1)
    return toks / ms  # tokens per ms


CLASSES = {"short": ["count", "story", "code", "science", "chat_short_rnone", "chat_short_r0.9", "chat_system_rnone",
                     "chat_system_r0.9", "chat_multiturn_rnone", "chat_multiturn_r0.9", "chat_reasoning_rnone", "chat_reasoning_r0.9"],
           "1k": ["len1k"], "4k": ["len4k"], "16k": ["len16k"], "prose": ["long_prose_300", "long_prose_700"], "code": ["code_tests"]}


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    tr, cost = parse(sys.argv[1]), fit(sys.argv[2])
    names = [p for p in tr if p in cost and p != "warm" and tr[p]]
    npass = sum(len(tr[p]) for p in names)
    mism = sum(1 for p in names for ps in tr[p] if not ps["match"])
    print(f"prompts {len(names)}, passes {npass}, conf/draft token mismatches {mism}")
    # check: k5's own trace vs min(A7, 5) + 1
    if len(sys.argv) > 3:
        t5 = parse(sys.argv[3])
        print("check tokens/pass, k5 trace vs sim from k7 (min(A,5)+1):")
        for p in names:
            if p in t5 and t5[p]:
                own = st.mean(ps["acc"] + 1 for ps in t5[p])
                sim = st.mean(min(ps["acc"], 5) + 1 for ps in tr[p])
                print(f"  {p:22s} {own:5.2f} {sim:5.2f}")
    # oracle: n = A (capped)
    for cap in (5, 7):
        r = []
        for p in names:
            a, b = cost[p]
            base = speed(tr[p], cost[p], None, 0, 5)
            orc = sum(min(ps["acc"], cap) + 1 for ps in tr[p]) / sum(a + b * (min(ps["acc"], cap) + 1) for ps in tr[p])
            r.append(orc / base)
        print(f"oracle cap {cap}: geo over prompts {geo(r):.3f}  (min {min(r):.3f} max {max(r):.3f})")
    # grid
    best = []
    print("\nsignal  cap  tau   | " + " ".join(f"{c:>6s}" for c in CLASSES) + " |  geo(all)  mean rows")
    have16 = all(not math.isnan(ps["conf"][0][5]) for p in names for ps in tr[p])
    for sig_name, sig in SIGNALS.items():
        if sig == 5 and not have16:
            continue
        for cap in (5, 6, 7):
            for tau in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
                per = {}
                for p in names:
                    per[p] = speed(tr[p], cost[p], sig, tau, cap) / speed(tr[p], cost[p], None, 0, 5)
                cls = {c: geo([per[p] for p in ps if p in per]) for c, ps in CLASSES.items() if any(p in per for p in ps)}
                g = geo(list(per.values()))
                rows = st.mean(kept(ps["conf"], sig, tau, cap) + 1 for p in names for ps in tr[p])
                best.append((g, sig_name, cap, tau, cls))
                print(f"{sig_name:7s} {cap:3d}  {tau:4.2f}  | " + " ".join(f"{cls.get(c, float('nan')):6.3f}" for c in CLASSES) + f" |  {g:6.3f}    {rows:4.2f}")
    best.sort(key=lambda x: -x[0])
    print("\nbest 5:")
    for g, s, c, t, cls in best[:5]:
        print(f"  {s} cap {c} tau {t}: geo {g:.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in cls.items()))


if __name__ == "__main__":
    main()
