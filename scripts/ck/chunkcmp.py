#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Lane ck: compare prefill self-golden dumps (`intarsia accuracy --write-self-golden`) written at different --prefill-chunk values.

    chunkcmp.py DIR [--ref 2048] [--json OUT]       # DIR holds prefill-c<chunk>.safetensors

Per chunk vs the reference chunk and per prompt: rows, rows that differ in any top-k id or any logprob bit, bitwise y/n,
max |d logprob| over the ids both rows share (a log-prob is logit - logsumexp, so with an unchanged logsumexp this IS |d logit|;
the dump keeps only top-k log-probs, not raw logits), max and mean per-row KL(ref || run) over the shared top-k ids
(renormalised), and argmax flips. "bitwise" means exactly 0 differing rows.
"""
import argparse, json, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "og-gates"))
from tailcmp import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", type=int, default=2048)
    ap.add_argument("--mode", default="prefill", help="prefill, or decode (teacher-forced single-token steps after a chunked prefill)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    pre = "" if a.mode == "prefill" else "dec_"
    files = sorted(d.glob(f"{a.mode}-c*.safetensors"), key=lambda p: int(re.search(r"-c(\d+)", p.name).group(1)))
    rp = d / f"{a.mode}-c{a.ref}.safetensors"
    if not rp.exists():
        raise SystemExit(f"reference {rp} missing")
    R, rmeta = load(rp)
    try:
        pm = json.loads(rmeta["prompts"])
    except Exception:
        pm = []
    n_prompts = len([k for k in R if k.startswith(f"{pre}topk_ids_")])
    report = []
    print(f"{d}: {a.mode}, reference chunk {a.ref}, {n_prompts} prompts")
    print(f"{'chunk':>6}{'prompt':<18}{'rows':>6}{'rows!=':>7}{'bitwise':>8}{'max|dlp|':>11}{'maxKL':>10}{'meanKL':>10}{'flips':>6}")
    for f in files:
        c = int(re.search(r"-c(\d+)", f.name).group(1))
        if c == a.ref:
            continue
        X, _ = load(f)
        for i in range(n_prompts):
            ri, rl, xi, xl = R[f"{pre}topk_ids_{i}"], R[f"{pre}topk_logprobs_{i}"], X[f"{pre}topk_ids_{i}"], X[f"{pre}topk_logprobs_{i}"]
            name = pm[i].get("name", str(i)) if i < len(pm) else str(i)
            n = ri.shape[0]
            if ri.shape != xi.shape:
                print(f"{c:>6}{name:<18} shape differs {ri.shape} vs {xi.shape}")
                continue
            neq = (ri != xi).any(axis=1) | (rl.view(np.int32) != xl.view(np.int32)).any(axis=1)
            maxd, kls = 0.0, np.full(n, np.nan)
            for r in range(n):
                common, ra, qa = np.intersect1d(ri[r], xi[r], return_indices=True)
                if len(common) == 0:
                    continue
                maxd = max(maxd, float(np.max(np.abs(rl[r][ra].astype(np.float64) - xl[r][qa].astype(np.float64)))))
                if len(common) >= 2:
                    p = np.exp(rl[r][ra].astype(np.float64)); p /= p.sum()
                    q = np.exp(xl[r][qa].astype(np.float64)); q /= q.sum()
                    kls[r] = float(np.sum(p * (np.log(p) - np.log(q))))
            flips = int((ri[np.arange(n), rl.argmax(axis=1)] != xi[np.arange(n), xl.argmax(axis=1)]).sum())
            row = dict(chunk=c, prompt=name, rows=n, rows_diff=int(neq.sum()), bitwise=bool(not neq.any()), max_abs_dlogprob=maxd,
                       max_kl=float(np.nanmax(kls)) if np.isfinite(kls).any() else float("nan"),
                       mean_kl=float(np.nanmean(kls)) if np.isfinite(kls).any() else float("nan"), flips=flips,
                       sum_kl=float(np.nansum(kls)), n_kl=int(np.isfinite(kls).sum()))
            report.append(row)
            print(f"{c:>6}{name:<18}{n:>6}{row['rows_diff']:>7}{'yes' if row['bitwise'] else 'NO':>8}{maxd:>11.3e}{row['max_kl']:>10.2e}{row['mean_kl']:>10.2e}{flips:>6}")
    print("aggregate over all prompts (row-weighted mean KL):")
    for c in sorted({r["chunk"] for r in report}):
        rs = [r for r in report if r["chunk"] == c]
        rows, rd = sum(r["rows"] for r in rs), sum(r["rows_diff"] for r in rs)
        nk = sum(r["n_kl"] for r in rs)
        print(f"  chunk {c}: rows {rows}, rows differing {rd}, max|dlp| {max(r['max_abs_dlogprob'] for r in rs):.3e}, "
              f"max KL {max(r['max_kl'] for r in rs):.2e}, mean KL {sum(r['sum_kl'] for r in rs) / max(nk, 1):.2e}, flips {sum(r['flips'] for r in rs)}")
    allb = all(r["bitwise"] for r in report)
    print(f"ALL BITWISE: {'yes' if allb and report else 'NO'}  (chunks compared: {sorted({r['chunk'] for r in report})})")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
