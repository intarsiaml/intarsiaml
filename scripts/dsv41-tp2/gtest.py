#!/usr/bin/env python3
"""Lane dsv41/sampled-spec: the sampling arm of the G-test, and its analysis.

    gtest.py run URL OUT.jsonl N SEED0 [--tokens 6] [--temp 1.0] [--prompt TEXT]
        N sequential /v1/completions requests (one short prompt, so each is a prefix hit and a few decode steps), `--tokens` tokens each at
        temperature --temp (top-k off, top-p 1), seed SEED0 + i; one line per request: {"seed", "ids"} (token ids of the reply).
    gtest.py test A.jsonl B.jsonl [--alpha 0.01] [--min-expected 5]
        G-test of homogeneity of the two arms' token frequencies at each position (1 .. tokens): the tokens' pooled counts rank them, the
        top bins keep their own cell while both arms' expected counts are at least --min-expected, the rest pool in "other". Bonferroni
        across positions. Also the power: the Cohen's w the test detects at power 0.8 with these counts (noncentral chi-square).
"""
import argparse
import http.client
import json
import sys
import urllib.parse


def run(a):
    u = urllib.parse.urlparse(a.url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
    with open(a.out, "w") as f:
        for i in range(a.n):
            body = {"model": "dsv41-intarsia", "prompt": a.prompt, "max_tokens": a.tokens, "temperature": a.temp, "top_p": 1.0, "seed": a.seed0 + i,
                    "ignore_eos": True, "return_token_ids": True}
            conn.request("POST", u.path.rstrip("/") + "/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = conn.getresponse()
            d = json.loads(r.read())
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {d}")
            ch = d["choices"][0]
            ids = ch.get("token_ids") or d.get("token_ids")
            f.write(json.dumps({"seed": a.seed0 + i, "ids": ids}) + "\n")
            if i % 200 == 0:
                print(f"{i}/{a.n}", flush=True)


def load(p):
    return [json.loads(l)["ids"] for l in open(p) if l.strip()]


def test(a):
    import numpy as np
    from scipy import stats

    A, B = load(a.a), load(a.b)
    L = min(min(map(len, A)), min(map(len, B)))
    out = []
    for pos in range(L):
        ca, cb = {}, {}
        for r in A:
            ca[r[pos]] = ca.get(r[pos], 0) + 1
        for r in B:
            cb[r[pos]] = cb.get(r[pos], 0) + 1
        toks = sorted(set(ca) | set(cb), key=lambda t: -(ca.get(t, 0) + cb.get(t, 0)))
        na, nb = len(A), len(B)
        rows, other = [], [0, 0]
        for t in toks:
            x, y = ca.get(t, 0), cb.get(t, 0)
            tot = x + y
            if tot * na / (na + nb) >= a.min_expected and tot * nb / (na + nb) >= a.min_expected:
                rows.append((x, y))
            else:
                other[0] += x
                other[1] += y
        if sum(other) > 0:
            rows.append(tuple(other))
        t = np.array(rows, dtype=float)
        df = len(rows) - 1
        g, p, _, _ = stats.chi2_contingency(t, correction=False, lambda_="log-likelihood")
        n = t.sum()
        alpha = a.alpha / L
        crit = stats.chi2.ppf(1 - alpha, df)
        # w detectable at power 0.8: noncentral chi-square with lambda = n w^2 (n = both arms' draws)
        lo, hi = 1e-4, 1.0
        for _ in range(60):
            w = (lo + hi) / 2
            pw = 1 - stats.ncx2.cdf(crit, df, n * w * w)
            lo, hi = (w, hi) if pw < 0.8 else (lo, w)
        out.append((pos + 1, len(rows), df, g, p, hi, alpha))
        print(f"position {pos + 1}: {len(rows)} cells (df {df}), G {g:.2f}, p {p:.4f} (alpha {alpha:.4f}: {'REJECT' if p < alpha else 'ok'}); detectable w at power 0.8: {hi:.3f}")
    print(f"N {len(A)} + {len(B)}; min p {min(o[4] for o in out):.4f}; all above alpha: {all(o[4] >= o[6] for o in out)}")
    return 0 if all(o[4] >= o[6] for o in out) else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("url"); r.add_argument("out"); r.add_argument("n", type=int); r.add_argument("seed0", type=int)
    r.add_argument("--tokens", type=int, default=6); r.add_argument("--temp", type=float, default=1.0)
    r.add_argument("--prompt", default="My favorite color is")
    t = sp.add_parser("test")
    t.add_argument("a"); t.add_argument("b"); t.add_argument("--alpha", type=float, default=0.01); t.add_argument("--min-expected", type=float, default=5)
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else test(a)


if __name__ == "__main__":
    sys.exit(main())
