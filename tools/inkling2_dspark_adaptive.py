"""Offline simulation of adaptive DSpark draft lengths (lane ink-mtp), from the oracle's draft logs.

Input: the `out` directory of `inkling2_dspark_ref.py draft` (per text: `taps_<name>.safetensors` with the target's greedy
argmax and the text ids, `draft_<name>.safetensors` with the 7 drafts and their top-1 probabilities at every anchor). For each
anchor t the accepted length m(t) of 7 drafts is the oracle's `match_len` (on the target's greedy path only).

A run is a trajectory: from the first anchor of a text, each cycle picks a draft length, accepts min(m(t), k_eff) tokens and
emits accepted + 1, so the next anchor is t + accepted + 1 (as the engine walks the text). Off-path anchors (vLLM's near-ties
against the W4A16 oracle) accept nothing. Ids are identical whatever k is: only the time moves.

Cost model (ms; `--fit` prints the check): a plain step 1000 / 62.58 (spec off, #10874). The drafter, from its measured profile
split (k3 tau 0.6, docs/results/2026-11-21-inkling2-mtp.md): block pass + ingest 2.5 per cycle, the head 0.40 per row it computes
(the cap k: the base logits of every draft row in one call), the Markov head 0.25 per step it runs (kept + 1 when the gate stops
early, k when it does not). A cycle that skips the drafter (k = 0) still ingests its row (0.7). Each extra verify row costs ROW,
the one parameter fitted (least squares) to the measured fixed policies k2 tau 0.6, k3 tau 0.6 and k3 tau 0.8; `--fit` prints the
residuals and the k4 / k3 ratio against #10806's (69.24 / 68.50 = 1.011, tau 0.6, an older binary).

(The first version, a flat 3.8 ms drafter per cycle and ROW 2.9, overrated the policies that draft more: k3 tau 0.6 84.2
modelled against 80.46 measured.)

Run from `tools/`: uv run python inkling2_dspark_adaptive.py --out <draft out dir> [--json results.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

from inkling2_dspark_ref import match_len

PLAIN, INGEST = 1000.0 / 62.58, 0.7
D_FIXED, D_HEAD_ROW, D_STEP = 2.5, 0.40, 0.25
ROW = 2.9  # replaced by the fit in main()


def load(out: Path) -> dict[str, list[dict]]:
    """Per text kind ('greedy', 'prose'): a list of texts, each {t: (m, p1[7])} over its anchors (m = -1 off the greedy path)."""
    sets: dict[str, list[dict]] = {}
    for dpath in sorted(out.glob("draft_*.safetensors")):
        name = dpath.stem[6:]
        d = load_file(str(dpath))
        tp = load_file(str(out / f"taps_{name}.safetensors"))
        kind = safe_open(str(out / f"taps_{name}.safetensors"), "pt").metadata()["kind"]
        ids, g = tp["ids"], tp["argmax"]
        recs = {}
        for a, t in enumerate(d["anchor_pos"].tolist()):
            on = int(ids[t]) == int(g[t - 1])
            m = match_len(d["draft"][a], g, ids, t, 7) if on else -1
            recs[t] = (m, [float(x) for x in d["p1"][a].tolist()])
        sets.setdefault(kind, []).append(recs)
    return sets


def gate(p1, k, tau):
    """Drafts the confidence gate keeps: the leading steps with top-1 probability >= tau, at most k."""
    n = 0
    while n < k and p1[n] >= tau:
        n += 1
    return n


def steps_run(n, cap):
    """Markov steps the drafter runs to keep n drafts under the cap: the step that fails the gate runs too."""
    return min(n + 1, cap)


def simulate(texts, policy) -> dict:
    """policy.choose(state, p1) -> (run_drafter, k_eff, cap); state is the policy's own dict, updated by policy.observe."""
    toks = ms = cycles = passes = drafted = 0
    for recs in texts:
        t = min(recs)
        st: dict = {}
        while t in recs:
            m, p1 = recs[t]
            run, k, cap = policy.choose(st, p1)
            acc = 0 if m < 0 else min(m, k)
            policy.observe(st, k, acc, run)
            if run:
                draft = D_FIXED + D_HEAD_ROW * cap + D_STEP * steps_run(k, cap)
            else:
                draft = 0.0 if getattr(policy, "off", False) else INGEST
            cost = PLAIN + draft + ROW * k
            toks += acc + 1
            ms += cost
            cycles += 1
            passes += k > 0
            drafted += k
            t += acc + 1
    return {"tok_s": toks / ms * 1e3, "tok_cycle": toks / cycles, "verify_passes": passes, "cycles": cycles,
            "tok_pass": (toks - (cycles - passes)) / max(passes, 1), "rows": 1 + drafted / max(passes, 1), "ms_tok": ms / toks}


class Off:
    """No speculation (no drafter, no ingest)."""

    off, label = True, "spec off"

    def choose(self, st, p1):
        return False, 0, 0

    def observe(self, st, k, acc, run):
        pass


class Fixed:
    """The engine today: the drafter every cycle, k drafts cut by the gate."""

    def __init__(self, k, tau):
        self.k, self.tau, self.label = k, tau, f"fixed k{k} tau {tau}"

    def choose(self, st, p1):
        return True, gate(p1, self.k, self.tau), self.k

    def observe(self, st, k, acc, run):
        pass


class Cumulative:
    """The drafter every cycle; draft j is kept while the product of the top-1 probabilities of drafts 1..j is >= c (an
    estimate of P(all j accepted)); a row pays when P x (ms per token, ~12.4) > 2.9 ms, i.e. c ~ 0.23."""

    def __init__(self, c, kmax=3):
        self.c, self.kmax, self.label = c, kmax, f"cumulative p1 >= {c}, kmax {kmax}"

    def choose(self, st, p1):
        n, prod = 0, 1.0
        while n < self.kmax:
            prod *= p1[n]
            if prod < self.c:
                break
            n += 1
        return True, n, self.kmax

    def observe(self, st, k, acc, run):
        pass


class Ema:
    """A running estimate a of the per-draft acceptance (EMA over every draft verified: accepted 1, the first rejected 0).
    Each cycle picks kmax in 0..K maximising (1 + a + ... + a^k) / cost(k), k = 0 skipping the drafter; the gate (tau) and, when
    given, the cumulative-p1 cut then shorten it. After `probe` cycles without the drafter, one cycle drafts k = 1."""

    def __init__(self, lam=0.1, kmax=3, tau=0.0, c=None, probe=8, prior=0.7):
        self.lam, self.kmax, self.tau, self.c, self.probe, self.prior = lam, kmax, tau, c, probe, prior
        cut = f", cumulative p1 >= {c}" if c is not None else ""
        self.label = f"EMA lambda {lam}, kmax {kmax}, tau {tau}{cut}"

    def choose(self, st, p1):
        a = st.setdefault("a", self.prior)
        best, bk = -1.0, 0
        for k in range(0, self.kmax + 1):
            e = 1 + sum(a ** j for j in range(1, k + 1))
            cost = PLAIN + (D_FIXED + (D_HEAD_ROW + D_STEP + ROW) * k if k else INGEST)
            if e / cost > best:
                best, bk = e / cost, k
        if bk == 0:
            st["idle"] = st.get("idle", 0) + 1
            if st["idle"] < self.probe:
                return False, 0, 0
            bk = 1
        st["idle"] = 0
        k = gate(p1, bk, self.tau)
        if self.c is not None:
            n, prod = 0, 1.0
            while n < k:
                prod *= p1[n]
                if prod < self.c:
                    break
                n += 1
            k = n
        return True, k, bk

    def observe(self, st, k, acc, run):
        a = st.get("a", self.prior)
        for j in range(min(k, acc + 1)):
            a = (1 - self.lam) * a + self.lam * (1.0 if j < acc else 0.0)
        st["a"] = a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    sets = load(args.out)
    # fit ROW to the measured fixed policies (#10874, greedy golden, tok/s)
    global ROW
    meas = [(Fixed(2, 0.6), 79.84), (Fixed(3, 0.6), 80.46), (Fixed(3, 0.8), 80.64)]
    best = None
    for r10 in range(150, 451):
        ROW = r10 / 100
        err = sum((simulate(sets["greedy"], p)["tok_s"] / m - 1) ** 2 for p, m in meas)
        if best is None or err < best[0]:
            best = (err, ROW)
    ROW = best[1]
    print(f"fit: ROW {ROW:.2f} ms per extra verify row")
    for p, m in meas:
        print(f"  {p.label}: modelled {simulate(sets['greedy'], p)['tok_s']:.2f}, measured {m}")
    k4 = simulate(sets["greedy"], Fixed(4, 0.6))["tok_s"] / simulate(sets["greedy"], Fixed(3, 0.6))["tok_s"]
    print(f"  k4 / k3 at tau 0.6: modelled {k4:.3f}, measured 1.011 (#10806, older binary)")
    pols = [Off(), Fixed(4, 0.6), Fixed(4, 0.8), Fixed(1, 0.6), Fixed(2, 0.6), Fixed(3, 0.6), Fixed(3, 0.8), Fixed(3, 0.9), Fixed(2, 0.8)]
    pols += [Cumulative(c, km) for km in (3, 4) for c in (0.15, 0.25, 0.35, 0.5)]
    pols += [Ema(lam, 3, tau) for lam in (0.05, 0.1, 0.2) for tau in (0.0, 0.6, 0.8)]
    pols += [Ema(lam, km, 0.0, c) for lam in (0.1, 0.2) for km in (3, 4) for c in (0.25, 0.35)]
    res = {}
    for kind in ("greedy", "prose"):
        if kind not in sets:
            continue
        base = simulate(sets[kind], Fixed(3, 0.8))
        res[kind] = []
        print(f"\n### {kind} ({len(sets[kind])} texts)\n")
        print("| policy | tok/s | vs fixed k3 tau 0.8 | tokens / cycle | tokens / verify pass | rows / pass | verify passes / cycles |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for p in pols:
            r = simulate(sets[kind], p)
            r["policy"], r["vs_base"] = p.label, r["tok_s"] / base["tok_s"]
            res[kind].append(r)
            print(f"| {p.label} | {r['tok_s']:.1f} | {r['vs_base']:.3f} | {r['tok_cycle']:.2f} | {r['tok_pass']:.2f} | {r['rows']:.2f} | "
                  f"{r['verify_passes']} / {r['cycles']} |")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
