#!/usr/bin/env python3
"""Lane ink-sampled-adapt: cost table and offline replay from W1's calibration runs (results doc section 3, pre-registered rule).

`sim.py CALDIR [--hip|--metal] [--grid] [--set key=v,...]`. CALDIR holds `<mode>-t<T>.err` from `generate --show-ids` with
INTARSIA_SPEC_TRACE=1 and INTARSIA_SPEC_SAMPLED_DFLASH=1: mode `0` (fixed k, INTARSIA_SPEC_SAMPLED_ADAPT=0) and `drafts` (every step a
draft call at the full cap, then a plain step), the same seed and prompts. Per run, per prompt: the `adapt: step` lines (kind, drafts,
tokens, ms), the `adapt: drafts` lines (the draft's proposals at every position: the replay's input) and `spec: keyed draft` times.

1. Cost table: D = median keyed draft ms (fixed run); cycle_m = median ms of fixed-run passes with m drafts (m with >= 10 passes);
   P = median drafts-run step ms - median drafts-run draft ms. d = D / P, r_m = (cycle_m - D) / P - 1. Compared with the built-in table:
   any of d, r_1..r_K more than 15 % off -> the measured table.
2. Replay check: fixed k replayed on the drafts run's proposals (accepted = the longest prefix equal to the emitted ids) must give the
   fixed run's passes / proposed / accepted per prompt exactly (the draft is deterministic given the context), and its predicted time is
   compared with the measured decode time.
3. Policy replay (a port of spec::AdaptK): predicted pol / off and pol / fix per prompt and class (off = every token a plain step P).
   --grid: g x margin x hold0 x hold_max of the pre-registered grid."""
import itertools
import math
import re
import statistics as st
import sys
from pathlib import Path

d = Path(sys.argv[1])
BACK = "metal" if "--metal" in sys.argv else "hip"
CLASSES = {"short": ["short_a", "short_b", "short_c"], "1k": ["len1k"], "4k": ["len4k"]}
PROMPTS = ["short_b", "short_a", "short_c", "len1k", "len4k"]
step_re = re.compile(r"^adapt: step (\w+) (\d+) (\d+) tokens ([0-9.]+) ms")
drafts_re = re.compile(r"^adapt: drafts at token (\d+) pos (\d+) cap (\d+) \[(.*)\]")
kd_re = re.compile(r"^spec: keyed draft ([0-9.]+) ms, (\d+) drafts")
ids_re = re.compile(r"^ids\[([^\]]*)\]: \[?(.*?)\]?$")
dec_re = re.compile(r"^decode\[([^\]]*)\]: (\d+) tokens, (\d+) ms")
spec_re = re.compile(r"^spec\[([^\]]*)\]: verify passes=(\d+) proposed=(\d+) accepted=(\d+)")


def parse(f):
    out, cur = {}, {"steps": [], "drafts": {}, "kd": []}
    for line in f.read_text(errors="replace").splitlines():
        if m := step_re.match(line):
            cur["steps"].append((m[1], int(m[2]), int(m[3]), float(m[4])))
        elif m := drafts_re.match(line):
            cur["drafts"][int(m[1]) - 1] = [int(x) for x in m[4].replace(",", " ").split()]
        elif m := kd_re.match(line):
            cur["kd"].append((float(m[1]), int(m[2])))
        elif m := ids_re.match(line):
            cur["ids"] = [int(x) for x in m[2].replace(",", " ").split()]
            out[m[1]] = cur
            cur = {"steps": [], "drafts": {}, "kd": []}
        elif m := dec_re.match(line):
            out[m[1]]["dec"] = (int(m[2]), int(m[3]))
        elif m := spec_re.match(line):
            out[m[1]]["spec"] = tuple(int(x) for x in m.groups()[1:])
    return out


def builtin(back):
    if back == "metal":
        return dict(d=0.24, r=[0.32, 0.51, 0.70], g=0.9, n0=1.0, a0=0.6, margin=0.02, drop=0.0, probe=2, hold0=8, hold_max=64, explore=8)
    return dict(d=0.26, r=[0.26, 0.47, 0.60, 0.75, 0.91], g=0.9, n0=1.0, a0=0.6, margin=0.02, drop=0.0, probe=2, hold0=8, hold_max=64, explore=8)


def row_cost(p, m):
    r = p["r"]
    n = len(r)
    if m == 0 or n == 0:
        return 0.0
    if m <= n:
        return r[m - 1]
    if n == 1:
        return r[0] * m
    return r[n - 1] + (r[n - 1] - r[n - 2]) * (m - n)


class AdaptK:
    """port of crates/intarsia-core/src/spec/adapt.rs AdaptK"""

    def __init__(self, p, kmax):
        self.p, self.kmax = p, max(1, min(kmax, 16))
        self.n = [0.0] * self.kmax
        self.q = [0.0] * self.kmax
        self.s = [0.0] * self.kmax
        self.phase, self.left, self.hold, self.calls, self.pleft = "draft", 0, p["hold0"], 0, 0
        self.events = []

    def est(self, j):
        i = j - 1
        w = self.n[i] + self.p["n0"]
        return (self.q[i] + self.p["n0"] * self.p["a0"] ** i) / w, (self.s[i] + self.p["n0"] * self.p["a0"] ** j) / w

    def predict(self, k):
        t, c = 1.0, self.p["d"] + 1.0
        for j in range(1, min(k, self.kmax) + 1):
            q, s = self.est(j)
            t += s
            c += q * (row_cost(self.p, j) - row_cost(self.p, j - 1))
        return t, c

    def best(self):
        b = (self.kmax, 0.0)
        for k in range(1, self.kmax + 1):
            t, c = self.predict(k)
            if t / c > b[1] + 1e-12:
                b = (k, t / c)
        return b

    def next(self, k):
        if k == 0:
            return 0
        if self.phase == "plain":
            if self.left > 0:
                self.left -= 1
                return 0
            self.phase, self.pleft = "probe", self.p["probe"]
            return k
        if self.phase == "probe":
            return k
        if self.p["explore"] > 0 and (self.calls + 1) % self.p["explore"] == 0:
            return k
        return min(self.best()[0], k)

    def record(self, cap, proposed, accepted):
        self.calls += 1
        g = self.p["g"]
        for i in range(min(cap, self.kmax)):
            self.n[i] = g * self.n[i] + 1
            self.q[i] = g * self.q[i] + (proposed > i)
            self.s[i] = g * self.s[i] + (accepted > i)
        k, rate = self.best()
        if self.phase == "draft":
            if rate < 1 - self.p["drop"]:
                self.phase, self.left = "plain", self.hold
                self.events.append("stop")
        elif self.phase == "probe":
            if self.pleft > 1:
                self.pleft -= 1
            elif rate >= 1 + self.p["margin"]:
                self.phase, self.hold = "draft", self.p["hold0"]
                self.events.append("resume")
            else:
                self.hold = min(self.hold * 2, self.p["hold_max"])
                self.phase, self.left = "plain", self.hold
                self.events.append("hold")


def replay(rec, K, cost, policy=None):
    """walk the emitted ids with the drafts-run proposals; returns time ms, passes, proposed, accepted, plain steps, events"""
    ids, drafts = rec["ids"], rec["drafts"]
    N = len(ids)
    g, ms, passes, prop, acc, plain = 1, 0.0, 0, 0, 0, 0
    a = AdaptK(policy, K) if policy else None
    while g < N:
        k = min(K, N - g - 1)
        if a:
            k = a.next(k)
        if k == 0:
            ms += cost["P"]
            g += 1
            plain += 1
            continue
        dr = drafts.get(g, [])[:k]
        if not dr:
            ms += cost["D"] + cost["P"]
            g += 1
            plain += 1
            if a:
                a.record(k, 0, 0)
            continue
        m = len(dr)
        x = 0
        while x < m and g + x < N and dr[x] == ids[g + x]:
            x += 1
        ms += cost["cyc"](m)
        passes, prop, acc = passes + 1, prop + m, acc + x
        if a:
            a.record(k, m, x)
        g += x + 1
    return ms, passes, prop, acc, plain, (a.events if a else [])


def med(v):
    return st.median(v) if v else float("nan")


runs = {}
for f in sorted(d.glob("*-t*.err")):
    mode, t = f.stem.split("-t")
    runs[(mode, t)] = parse(f)
temps = sorted({t for _, t in runs}, reverse=True)
K = 3 if BACK == "metal" else 5
table = builtin(BACK)

# 1. cost table, pooled over the temperatures (and cards)
kd_fix, cyc, step_dr, kd_dr = [], {}, [], []
for (mode, t), recs in runs.items():
    for name, r in recs.items():
        if name == "warm":
            continue
        if mode == "0":
            kd_fix += [x for x, n in r["kd"]]
            for kind, n, tok, ms in r["steps"]:
                if kind == "pass":
                    cyc.setdefault(n, []).append(ms)
        elif mode == "drafts":
            kd_dr += [x for x, n in r["kd"]]
            step_dr += [ms for kind, n, tok, ms in r["steps"] if kind in ("drafts", "empty")]
D = med(kd_fix)
P = med(step_dr) - med(kd_dr)
if not step_dr:
    # no drafts run (Metal slot): P from the fixed run's empty calls (a draft call that came back empty, then a plain step)
    emp = [ms for (mode, t), recs in runs.items() if mode == "0" for name, r in recs.items() if name != "warm"
           for kind, n, tok, ms in r["steps"] if kind == "empty"]
    kd0 = [x for (mode, t), recs in runs.items() if mode == "0" for name, r in recs.items() if name != "warm" for x, n in r["kd"] if n == 0]
    P = med(emp) - med(kd0)
    print(f"no drafts run: P from {len(emp)} empty calls of the fixed run: {med(emp):.3f} - {med(kd0):.3f} ms")
meas = {"d": D / P, "r": []}
print(f"backend {BACK}, K {K}; runs {sorted(runs)}")
print(f"D (keyed draft, fixed run) median {D:.3f} ms (n {len(kd_fix)}); drafts run: step median {med(step_dr):.3f} ms, draft median "
      f"{med(kd_dr):.3f} ms -> P {P:.3f} ms")
for m in sorted(cyc):
    v = cyc[m]
    rm = (med(v) - D) / P - 1
    print(f"  pass with {m} drafts: n {len(v)}, cycle median {med(v):.3f} ms -> r_{m} = {rm:.3f}" + ("" if len(v) >= 10 else " (n < 10: not used)"))
    if len(v) >= 10 and m <= K:
        meas["r"].append((m, rm))
off = []
if abs(meas["d"] - table["d"]) > 0.15 * table["d"]:
    off.append(f"d {meas['d']:.3f} vs {table['d']}")
for m, rm in meas["r"]:
    if m <= len(table["r"]) and abs(rm - table["r"][m - 1]) > 0.15 * table["r"][m - 1]:
        off.append(f"r_{m} {rm:.3f} vs {table['r'][m - 1]}")
print(f"built-in d {table['d']}, r {table['r']}; measured d {meas['d']:.3f}, r " + " ".join(f"{m}:{x:.3f}" for m, x in meas["r"]))
print("15 % rule: " + ("built-in table stays" if not off else "MEASURED TABLE (" + "; ".join(off) + ")"))
mt = dict(table)
if off:
    rs = dict(meas["r"])
    # rows without enough passes keep the built-in value scaled by the measured rows' mean ratio
    sc = st.mean(rs[m] / table["r"][m - 1] for m in rs if m <= len(table["r"])) if rs else 1.0
    mt["d"] = round(meas["d"], 3)
    mt["r"] = [round(rs.get(m, table["r"][m - 1] * sc), 3) for m in range(1, len(table["r"]) + 1)]
    print(f"measured table: d={mt['d']},r={'/'.join(str(x) for x in mt['r'])}")


def cyc_ms(m):
    v = cyc.get(m, [])
    return med(v) if len(v) >= 3 else D + P * (1 + row_cost(mt, m))


cost = {"D": D, "P": P, "cyc": cyc_ms}

# 2. replay check
print("\nreplay of fixed k on the drafts run's proposals vs the fixed run (passes / proposed / accepted; ms predicted vs decode ms):")
okall = True
for t in temps:
    fx, dr = runs.get(("0", t)), runs.get(("drafts", t))
    if not fx or not dr:
        continue
    for name in ["warm"] + PROMPTS:
        if name not in fx or name not in dr:
            continue
        if fx[name]["ids"] != dr[name]["ids"]:
            print(f"  T {t} {name}: ids differ between the fixed and drafts runs")
            okall = False
            continue
        ms, pa, pr, ac, pl, _ = replay(dr[name], K, cost)
        sp = fx[name].get("spec")
        same = sp == (pa, pr, ac)
        okall &= same
        dm = fx[name]["dec"][1]
        print(f"  T {t} {name:8s} replay {pa}/{pr}/{ac} measured {sp[0]}/{sp[1]}/{sp[2]} {'same' if same else 'DIFFERENT'}; "
              f"{ms:.0f} ms predicted vs {dm} measured ({ms / dm:.3f})")
print("replay check: " + ("exact" if okall else "NOT exact"))


TABLE = f"d={mt['d']},r={'/'.join(str(x) for x in mt['r'])}" if off else ""
if not any(m == "drafts" for m, _ in runs):
    print(f"adapt string: {TABLE}")
    sys.exit(0)

# 3. policy replay
def evaluate(pol, show=False):
    res = {}
    for t in temps:
        dr = runs.get(("drafts", t))
        if not dr:
            continue
        for name in PROMPTS:
            if name not in dr:
                continue
            r = dr[name]
            n = len(r["ids"]) - 1
            t_off = n * P
            t_fix = replay(r, K, cost)[0]
            ms, pa, pr, ac, pl, ev = replay(r, K, cost, pol)
            res[(t, name)] = (t_off / t_fix, t_off / ms, t_fix / ms, pa, ac, pl, ev)
    if show:
        print("| T | prompt | fix / off | pol / off | pol / fix | pol passes, accepted, plain steps | stops / holds / resumes |")
        print("|---|---|---:|---:|---:|---|---|")
        for (t, name), (f, p, pf, pa, ac, pl, ev) in res.items():
            print(f"| {t} | {name} | {f:.3f} | {p:.3f} | {pf:.3f} | {pa}, {ac}, {pl} | {ev.count('stop')} / {ev.count('hold')} / {ev.count('resume')} |")
    return res


def classes(res):
    out = {}
    for t in temps:
        for cls, names in CLASSES.items():
            v = [res[(t, n)] for n in names if (t, n) in res]
            if v:
                # pooled over the class's prompts: same token count per prompt, so the harmonic mean of the ratios
                f = len(v) / sum(1 / x[0] for x in v)
                p = len(v) / sum(1 / x[1] for x in v)
                out[(t, cls)] = (f, p)
    return out


print(f"\npolicy replay, table d={mt['d']}, r={mt['r']} (other parameters built-in):")
base = dict(mt)
for kv in next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--set=")), "").split(","):
    if "=" in kv:
        k, v = kv.split("=")
        base[k] = [float(x) for x in v.split("/")] if k == "r" else (int(v) if k in ("hold0", "hold_max", "explore", "probe") else float(v))
res = evaluate(base, show=True)
cl = classes(res)
for (t, cls), (f, p) in cl.items():
    print(f"  T {t} {cls}: fix / off {f:.3f}, pol / off {p:.3f}, pol / (0.99 x fix) {p / (0.99 * f):.3f}")
mn = min(v[1] for v in res.values())
bad = [k for k, v in res.items() if v[1] < 0.99] + [k for k, (f, p) in cl.items() if p < 0.99 * f]
print(f"  lowest prompt pol / off {mn:.3f}; " + ("defaults meet the 0.99 bars" if not bad else f"defaults miss the 0.99 bars at {bad}: grid"))

if "--grid" in sys.argv or bad:
    best = None
    for g, mg, dp, pr, h0, hm in itertools.product([0.85, 0.9, 0.95], [0.0, 0.02, 0.05], [0.0, 0.05, 0.1], [1, 2, 3], [8, 16], [32, 64]):
        pp = dict(base, g=g, margin=mg, drop=dp, probe=pr, hold0=h0, hold_max=hm)
        r = evaluate(pp)
        c = classes(r)
        keep = all(p >= 0.99 * f for f, p in c.values())
        mn = min(v[1] for v in r.values())
        if keep and (best is None or mn > best[0]):
            best = (mn, pp)
        if "--grid-all" in sys.argv:
            print(f"  g {g} margin {mg} drop {dp} probe {pr} hold {h0} hold_max {hm}: min prompt {mn:.3f}, classes keep 0.99 x fix: {keep}")
    GRID = ""
    if best:
        b = best[1]
        GRID = f"g={b['g']},margin={b['margin']},drop={b['drop']},probe={b['probe']},hold={b['hold0']},hold_max={b['hold_max']}"
        print(f"grid best: min prompt {best[0]:.3f} at ADAPT=g={b['g']},margin={b['margin']},drop={b['drop']},probe={b['probe']},hold={b['hold0']},hold_max={b['hold_max']}")
        evaluate(b, show=True)
        for (t, cls), (f, p) in classes(evaluate(b)).items():
            print(f"  T {t} {cls}: fix / off {f:.3f}, pol / off {p:.3f}")
    else:
        print("grid best: none keeps every class")

print("adapt string: " + ",".join(x for x in [TABLE, globals().get("GRID", "")] if x))
