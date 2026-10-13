#!/usr/bin/env python3
"""CPU-reference gate of docs/results/2026-10-14-35b-accuracy.md section 2.2, offline from dumps.

    uv run --project tools python scripts/acc35b/cpuref_gate.py --ref cpu=DUMP --env oldmoe=D simt=D hip=D \
        --test cuda=D mut105=D ... [--k 2]

Statistics per mode are those of `intarsia accuracy --self-golden` (selfgolden::judge): mean KL21(ref || X), largest
single-position KL21, top-1 flips, mean |tail diff| (tail = 1 - mass on the reference's top-20 ids), worst prompt's mean
KL21. Budget per mode and statistic = k x max over the envelope implementations; a test run passes when every statistic
is within budget (flips: at least one allowed, as in the tool). Prints markdown tables and the budget flags for the tool
(the tool takes one value per flag for both modes: the larger of the two modes' budgets).
"""
import argparse, math
import numpy as np
from xcompare import load, TAIL_K


def stats(ref, cand, mode):
    ks, flips, tails, per = [], 0, [], {}
    for pn, r in ref.items():
        rows = r["dec"] if mode == "decode" else r["pre"]
        crow = cand[pn]["dec"] if mode == "decode" else cand[pn]["pre"]
        if not np.array_equal(r["ids"], cand[pn]["ids"]):
            raise SystemExit(f"{pn}: ids differ")
        pk = []
        for pos in sorted(rows):
            rid, rlp = rows[pos]
            cid, clp = crow[pos]
            m = rid >= 0
            rid, rlp = rid[m][:TAIL_K], rlp[m][:TAIL_K].astype(np.float64)
            cm = cid >= 0
            cmap = dict(zip(cid[cm].tolist(), clp[cm].astype(np.float64).tolist()))
            floor = float(clp[cm].min())
            pr = np.exp(rlp)
            pc = np.array([math.exp(cmap.get(t, floor)) for t in rid.tolist()])
            tr, tc = max(1 - pr.sum(), 1e-12), max(1 - pc.sum(), 1e-12)
            k = float(np.sum(pr * (rlp - np.log(np.maximum(pc, 1e-30)))) + tr * (math.log(tr) - math.log(tc)))
            ks.append(max(k, 0.0)); pk.append(max(k, 0.0)); tails.append(abs(tr - tc))
            flips += int(cid[cm][np.argmax(clp[cm])]) != int(rid[np.argmax(rlp)])
        per[pn] = float(np.mean(pk))
    n = len(ks)
    return {"n": n, "mean_kl21": float(np.mean(ks)), "max_kl21": float(np.max(ks)), "flips": flips,
            "flip_pct": 100.0 * flips / n, "tail_abs": float(np.mean(tails)), "worst": max(per.values()),
            "worst_name": max(per, key=per.get)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--env", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--k", type=float, default=2.0)
    a = ap.parse_args()
    rn, rp = a.ref.split("=", 1)
    ref = load(rp)[0]
    env = [(s.split("=", 1)[0], load(s.split("=", 1)[1])[0]) for s in a.env]
    test = [(s.split("=", 1)[0], load(s.split("=", 1)[1])[0]) for s in a.test]
    keys = ["mean_kl21", "max_kl21", "flip_pct", "tail_abs", "worst"]
    flags = {}
    for mode in ("prefill", "decode"):
        S = {n: stats(ref, d, mode) for n, d in env + test}
        budget = {k: a.k * max(S[n][k] for n, _ in env) for k in keys}
        for k in keys:
            flags[k] = max(flags.get(k, 0.0), budget[k])
        print(f"\n{mode} (n = {S[env[0][0]]['n']} positions), KL21({rn} || X)\n")
        print("| X | role | mean KL21 | max pos KL21 | flips (%) | mean abs tail diff | worst prompt mean KL21 | verdict |")
        print("|---|---|---|---|---|---|---|---|")
        for n, _ in env + test:
            s = S[n]
            role = "envelope" if any(n == e for e, _ in env) else "judged"
            fl_lim = max(budget["flip_pct"] / 100 * s["n"], 1.0)
            fails = [k for k in ("mean_kl21", "max_kl21", "tail_abs", "worst") if s[k] > budget[k]]
            if s["flips"] > math.floor(fl_lim + 1e-9):
                fails.append("flips")
            v = "-" if role == "envelope" else ("PASS" if not fails else "FAIL (" + ", ".join(fails) + ")")
            print(f"| {n} | {role} | {s['mean_kl21']:.3e} | {s['max_kl21']:.3f} | {s['flips']} ({s['flip_pct']:.2f}) | "
                  f"{s['tail_abs']:.2e} | {s['worst']:.3e} ({s['worst_name']}) | {v} |")
        print(f"| **budget** (k = {a.k:g} x envelope max) | | {budget['mean_kl21']:.3e} | {budget['max_kl21']:.3f} | "
              f"{budget['flip_pct']:.2f} % | {budget['tail_abs']:.2e} | {budget['worst']:.3e} | |")
    print(f"\ntool flags (larger of the two modes): --self-max-kl {flags['mean_kl21']:.4e} --self-max-pos-kl {flags['max_kl21']:.4f} "
          f"--self-max-flip-pct {flags['flip_pct']:.4f} --self-max-tail {flags['tail_abs']:.4e} "
          f"--self-prompt-factor {flags['worst'] / flags['mean_kl21']:.4f}")


if __name__ == "__main__":
    main()
