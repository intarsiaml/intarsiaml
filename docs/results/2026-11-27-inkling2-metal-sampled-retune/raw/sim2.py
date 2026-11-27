#!/usr/bin/env python3
"""Lane ink-metal-sampled2: offline replay of the adaptive draft-length policy on `drafts` traces of several seeds, and the pre-registered
stage-A grid (results doc section 1.3). Port of crates/intarsia-core/src/spec/adapt.rs AdaptK, with optional extra rules (stage B knobs,
default off = today's behaviour exactly).

  sim2.py DRAFTSDIR [--po X] [--set k=v,...] [--grid A|B] [--eval "g=..,drop=.."] [--check TRACEFILE] [--loso] [--top N]

DRAFTSDIR holds `s<seed>-t<T>.err` from `generate --show-ids` with INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=drafts
INTARSIA_SPEC_TRACE=1 (every step a draft call at the full cap k3 and then a plain step, the proposals logged at every position).
Time model, in units of a plain step P (off = 1 per token): a pass with m drafts costs d + 1 + r_m; an empty draft call d + 1 (+ po for the plain
step it ends in); a plain step of the policy mode 1 + po. Cost table: measured by lane ink-sampled-adapt's Metal slot (results doc 2026-11-26)."""
import itertools
import math
import re
import sys
from pathlib import Path

TABLE = dict(d=0.237, r=[0.236, 0.418, 0.583])
BASE = dict(TABLE, g=0.9, n0=1.0, a0=0.6, margin=0.02, drop=0.0, probe=2, hold0=8, hold_max=64, explore=8, stop_n=1, hold_grow=2.0, settle=0, giveup=0)
K = 3
CLASSES = {"short": ["short_a", "short_b", "short_c"], "1k": ["len1k"], "4k": ["len4k"]}
PROMPTS = ["short_b", "short_a", "short_c", "len1k", "len4k"]
KEYMAP = {"hold": "hold0"}
ALIAS = {"hold0": "hold"}  # the recipe / env key names (AdaptParams::apply): hold, not hold0

step_re = re.compile(r"^adapt: step (\w+) (\d+) (\d+) tokens ([0-9.]+) ms")
drafts_re = re.compile(r"^adapt: drafts at token (\d+) pos (\d+) cap (\d+) \[(.*)\]")
ids_re = re.compile(r"^ids\[([^\]]*)\]: \[?(.*?)\]?$")
dec_re = re.compile(r"^decode\[([^\]]*)\]: (\d+) tokens, (\d+) ms")
spec_re = re.compile(r"^spec\[([^\]]*)\]: verify passes=(\d+) proposed=(\d+) accepted=(\d+)")


def parse(f):
    out, cur = {}, {"steps": [], "drafts": {}}
    for line in Path(f).read_text(errors="replace").splitlines():
        if m := step_re.match(line):
            cur["steps"].append((m[1], int(m[2]), int(m[3]), float(m[4])))
        elif m := drafts_re.match(line):
            cur["drafts"][int(m[1]) - 1] = [int(x) for x in m[4].replace(",", " ").split()]
        elif m := ids_re.match(line):
            cur["ids"] = [int(x) for x in m[2].replace(",", " ").split()]
            out[m[1]] = cur
            cur = {"steps": [], "drafts": {}}
        elif m := dec_re.match(line):
            out[m[1]]["dec"] = (int(m[2]), int(m[3]))
        elif m := spec_re.match(line):
            out[m[1]]["spec"] = tuple(int(x) for x in m.groups()[1:])
    return out


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
    """port of AdaptK; stop_n / hold_grow / settle are the stage-B knobs (defaults 1 / 2 / 0 = today's behaviour)"""

    def __init__(self, p, kmax):
        self.p, self.kmax = p, max(1, min(kmax, 16))
        self.n = [0.0] * self.kmax
        self.q = [0.0] * self.kmax
        self.s = [0.0] * self.kmax
        self.phase, self.left, self.hold, self.calls, self.pleft = "draft", 0, p["hold0"], 0, 0
        self.low = 0
        self.since_resume = 10**9
        self.nholds = 0
        self.gave_up = False
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
        if k == 0 or self.gave_up:
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
        self.since_resume += 1
        g = self.p["g"]
        for i in range(min(cap, self.kmax)):
            self.n[i] = g * self.n[i] + 1
            self.q[i] = g * self.q[i] + (proposed > i)
            self.s[i] = g * self.s[i] + (accepted > i)
        k, rate = self.best()
        if self.phase == "draft":
            if rate < 1 - self.p["drop"]:
                self.low += 1
                if self.low >= self.p["stop_n"] and self.since_resume >= self.p["settle"]:
                    self.phase, self.left, self.low = "plain", self.hold, 0
                    self.events.append("stop")
            else:
                self.low = 0
        elif self.phase == "probe":
            if self.pleft > 1:
                self.pleft -= 1
            elif rate >= 1 + self.p["margin"]:
                self.phase, self.hold, self.since_resume, self.low = "draft", self.p["hold0"], 0, 0
                self.events.append("resume")
            else:
                self.hold = min(int(self.hold * self.p["hold_grow"]), self.p["hold_max"])
                self.phase, self.left = "plain", self.hold
                self.events.append("hold")
                self.nholds += 1
                if self.p["giveup"] and self.nholds >= self.p["giveup"]:
                    self.gave_up = True
                    self.events.append("giveup")


def replay(rec, policy, po=0.0):
    """walk the emitted ids with the drafts-run proposals; time in plain-step units. policy None: fixed k (never a plain step by choice)"""
    ids, drafts = rec["ids"], rec["drafts"]
    N = len(ids)
    g, t, passes, prop, acc, plain = 1, 0.0, 0, 0, 0, 0
    d = (policy or BASE)["d"]
    a = AdaptK(policy, K) if policy else None
    pp = policy or BASE
    while g < N:
        k = min(K, N - g - 1)
        if a:
            k = a.next(k)
        if k == 0:
            t += 1.0 + (po if a and not a.gave_up else 0.0)
            g += 1
            plain += 1
            continue
        dr = drafts.get(g, [])[:k]
        if not dr:
            t += d + 1.0 + (po if a else 0.0)
            g += 1
            plain += 1
            if a:
                a.record(k, 0, 0)
            continue
        m = len(dr)
        x = 0
        while x < m and g + x < N and dr[x] == ids[g + x]:
            x += 1
        t += d + 1.0 + row_cost(pp, m)
        passes, prop, acc = passes + 1, prop + m, acc + x
        if a:
            a.record(k, m, x)
        g += x + 1
    return t, N - 1, (passes, prop, acc, plain), (a.events if a else [])


def load(d):
    runs = {}
    for f in sorted(Path(d).glob("s*-t*.err")):
        s, t = f.stem[1:].split("-t")
        runs[(int(s), t)] = parse(f)
    return runs


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def evaluate(runs, pol, po, seeds=None):
    """-> {(T, prompt): geomean pol/off}, {(T, class): (pol/off, fix/off)}, per-seed detail"""
    P, C, det = {}, {}, {}
    for t in sorted({t for _, t in runs}):
        ss = [s for (s, tt) in runs if tt == t and (seeds is None or s in seeds)]
        for nm in PROMPTS:
            v = []
            for s in ss:
                rec = runs[(s, t)].get(nm)
                if rec:
                    tp, n, _, _ = replay(rec, pol, po)
                    v.append(n / tp)
            if v:
                P[(t, nm)] = gm(v)
        for cls, names in CLASSES.items():
            vp, vf = [], []
            for s in ss:
                n_tot = tp_tot = tf_tot = 0.0
                for nm in names:
                    rec = runs[(s, t)].get(nm)
                    if not rec:
                        continue
                    tp, n, _, _ = replay(rec, pol, po)
                    tf, _, _, _ = replay(rec, None, 0.0)
                    n_tot, tp_tot, tf_tot = n_tot + n, tp_tot + tp, tf_tot + tf
                if n_tot:
                    vp.append(n_tot / tp_tot)
                    vf.append(n_tot / tf_tot)
            if vp:
                C[(t, cls)] = (gm(vp), gm(vf))
    return P, C


def score(P, C):
    s1 = min(v / 0.98 for v in P.values())
    s2 = min(p / (0.98 * f) for p, f in C.values())
    return min(s1, s2), s1, s2


def mean_cells(P, C):
    return sum(P.values()) / len(P)


def fmt(p):
    return ",".join(f"{ALIAS.get(k, k)}={p[k]}" for k in ("g", "margin", "drop", "probe", "hold0", "hold_max")) + (
        f",stop_n={p['stop_n']}" if p["stop_n"] != 1 else "") + (f",hold_grow={p['hold_grow']}" if p["hold_grow"] != 2.0 else "") + (
        f",settle={p['settle']}" if p["settle"] else "") + (f",giveup={p['giveup']}" if p["giveup"] else "")


def show(runs, pol, po, label=""):
    P, C = evaluate(runs, pol, po)
    print(f"{label} policy {fmt(pol)} po {po}")
    print("  per prompt pol/off (geomean over seeds):")
    for t in sorted({k[0] for k in P}):
        print(f"    T {t}: " + "  ".join(f"{nm} {P[(t, nm)]:.3f}" for nm in PROMPTS if (t, nm) in P))
    for t in sorted({k[0] for k in C}):
        print(f"    T {t} classes pol/off | fix/off | pol/(0.98 fix): " + "  ".join(
            f"{c} {C[(t, c)][0]:.3f}|{C[(t, c)][1]:.3f}|{C[(t, c)][0] / (0.98 * C[(t, c)][1]):.3f}" for c in CLASSES))
    sc = score(P, C)
    print(f"  score {sc[0]:.4f} (prompts {sc[1]:.4f}, classes {sc[2]:.4f})")
    return sc


def setkv(p, s):
    p = dict(p)
    for kv in s.split(","):
        if "=" in kv:
            k, v = kv.split("=")
            k = KEYMAP.get(k, k)
            p[k] = [float(x) for x in v.split("/")] if k == "r" else (int(v) if k in ("hold0", "hold_max", "explore", "probe", "stop_n", "settle", "giveup") else float(v))
    return p


def grid(stage):
    if stage == "A":
        return itertools.product([0.85, 0.9, 0.95, 0.97], [0.0, 0.02, 0.05], [0.0, 0.05, 0.1, 0.15, 0.2], [1, 2, 3], [8, 16, 32], [32, 64, 128],
                                 [1], [2.0], [0], [0])
    return itertools.product([0.9, 0.95, 0.97], [0.0, 0.02], [0.0, 0.05, 0.1], [1, 2], [8, 16], [64, 128], [1, 2, 3], [2.0, 4.0], [0], [0, 1, 2, 3])


def run_grid(runs, po, stage, seeds=None, top=8):
    res = []
    for g, mg, dp, pr, h0, hm, sn, hg, st, gu in grid(stage):
        p = dict(BASE, g=g, margin=mg, drop=dp, probe=pr, hold0=h0, hold_max=hm, stop_n=sn, hold_grow=hg, settle=st, giveup=gu)
        P, C = evaluate(runs, p, po, seeds)
        sc = score(P, C)
        res.append((sc[0], mean_cells(P, C), p))
    res.sort(key=lambda x: -x[0])
    best = res[0][0]
    near = [r for r in res if r[0] >= best - 0.003]
    pick = max(near, key=lambda r: r[1])
    return pick, res[:top], len(res)


if __name__ == "__main__":
    args = sys.argv[1:]
    runs = load(args[0])
    arg = lambda name, dflt=None: next((a.split("=", 1)[1] for a in args if a.startswith(name + "=")), dflt)
    po = float(arg("--po", 0.0))
    base = setkv(BASE, arg("--set", ""))
    print(f"traces: {sorted(runs)}; cost table d={TABLE['d']} r={TABLE['r']}; po {po}")
    # replay check: fixed k3 on the traces vs the traced fixed run
    chk = arg("--check")
    if chk:
        fx = parse(chk)
        s, t = int(arg("--check-seed", 13)), arg("--check-t", "1.0")
        print(f"replay check vs {chk} (seed {s}, T {t}): passes/proposed/accepted, predicted vs measured ms (units * P)")
        P_ms = float(arg("--P", 13.62))
        for nm in PROMPTS:
            if nm in fx and (s, t) in runs and nm in runs[(s, t)]:
                tp, n, (pa, pr, ac, pl), _ = replay(runs[(s, t)][nm], None)
                sp = fx[nm].get("spec")
                print(f"  {nm:8s} replay {pa}/{pr}/{ac} measured {sp[0]}/{sp[1]}/{sp[2]}; {tp * P_ms:.0f} ms vs {fx[nm]['dec'][1]} ({tp * P_ms / fx[nm]['dec'][1]:.3f})")
    if arg("--eval"):
        show(runs, setkv(base, arg("--eval")), po, "eval")
    show(runs, setkv(BASE, "g=0.85,margin=0.0,drop=0.0,probe=3,hold=8,hold_max=32"), po, "current recipe")
    show(runs, dict(BASE, g=0.95, margin=0.0, drop=0.05, probe=1, hold0=16, hold_max=64), po, "HIP recipe")
    if arg("--grid"):
        st = arg("--grid")
        pick, top, n = run_grid(runs, po, st)
        print(f"\nstage {st} grid: {n} settings; top by score:")
        for sc, mc, p in top:
            print(f"  score {sc:.4f} mean {mc:.4f}  {fmt(p)}")
        print("pick (best score, ties within 0.003 by mean pol/off):")
        show(runs, pick[2], po, "PICK")
        if "--loso" in args:
            seeds = sorted({s for s, _ in runs})
            print("leave-one-seed-out (pick on the others, evaluate on the held-out seed; score of the held-out cells):")
            for s in seeds:
                others = [x for x in seeds if x != s]
                pk, _, _ = run_grid(runs, po, st, others)
                P, C = evaluate(runs, pk[2], po, [s])
                print(f"  held out {s}: picked {fmt(pk[2])}; score on {s}: {score(P, C)[0]:.4f}")
        print("adapt string: d=%s,r=%s,%s" % (TABLE["d"], "/".join(str(x) for x in TABLE["r"]), fmt(pick[2])))
