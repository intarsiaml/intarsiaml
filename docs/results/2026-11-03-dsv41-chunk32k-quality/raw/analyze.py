#!/usr/bin/env python3
"""Paired analysis of the chunk32k run (docs/results/2026-11-03-dsv41-chunk32k-quality.md).

Reads localeval result.json files (label dsv41-chunk32k-ab<i>-<shard>), pairs the arms by case name, and writes
  <out>/analysis.md   tables (accuracy, discordant pairs, exact McNemar, Newcombe 95% CI, by kind and depth, truncation, text)
  <out>/replies-ab<i>.jsonl   per item: name, expected, reply, success, prompt/cached/completion tokens
Usage: python3 analyze.py --results /opt/code/deepeval/results --arms ab0=512 ab1=2048 [ab2=512-hc0] --since 20261103-1400 --out DIR
"""
import argparse, glob, json, math, os, re
from collections import defaultdict


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_paired(a, b, c, d, z=1.959964):
    """Newcombe (1998) method 10: CI of p1 - p2 for paired proportions. a both right, b only first, c only second, d neither."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    D = p1 - p2
    lo = D - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = D + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return D, lo, hi


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(results, arm, since):
    out = {}
    for f in sorted(glob.glob(f"{results}/*-adhoc-dsv41-chunk32k-{arm}-*/result.json")):
        if os.path.basename(os.path.dirname(f)) < since:
            continue
        r = json.load(open(f))
        for d in r["datasets"]:
            for c in d["cases"]:
                out[c["name"]] = {**c, "run_id": r["run_id"]}
    return out


def kind_of(name):
    return name.rsplit("-", 2)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/opt/code/deepeval/results")
    ap.add_argument("--arms", nargs="+", required=True, help="ab0=512 ab1=2048 ...; the first is the reference")
    ap.add_argument("--since", default="00000000", help="ignore result dirs older than this prefix (YYYYMMDD-HHMM)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = {m["name"]: m for m in json.load(open(a.manifest))["items"]}
    arms = [x.split("=") for x in a.arms]
    data = {lab: load(a.results, key, a.since) for key, lab in arms}
    for (key, lab) in arms:
        with open(f"{a.out}/replies-{key}.jsonl", "w") as f:
            for n in sorted(data[lab], key=lambda n: man[n]["idx"]):
                c = data[lab][n]
                f.write(json.dumps({"name": n, "expected": c["expected_output"], "reply": c["actual_output"], "success": bool(c["success"]),
                                    "prompt_tokens": c["speed"]["prompt_tokens"], "cached_tokens": c["speed"]["cached_tokens"],
                                    "completion_tokens": c["speed"]["completion_tokens"], "ttft_s": round(c["speed"]["ttft_s"], 2) if c["speed"]["ttft_s"] else None,
                                    "error": c["error"], "run_id": c["run_id"]}, ensure_ascii=False) + "\n")
    L = []
    L.append("### Accuracy per arm\n")
    L.append("| arm | items | correct | accuracy | Wilson 95%% CI | truncated replies (>= %d tokens) | call errors | cached tokens (sum / max per item) | prompt tokens (min-max) |" % a.max_tokens)
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key, lab in arms:
        d = data[lab]
        k = sum(bool(c["success"]) for c in d.values())
        lo, hi = wilson(k, len(d))
        tr = sum(1 for c in d.values() if (c["speed"]["completion_tokens"] or 0) >= a.max_tokens)
        er = sum(1 for c in d.values() if c["error"])
        ct = [c["speed"]["cached_tokens"] or 0 for c in d.values()]
        pt = [c["speed"]["prompt_tokens"] for c in d.values()]
        L.append(f"| {lab} ({key}) | {len(d)} | {k} | {k / max(len(d), 1):.3f} | {lo:.3f} to {hi:.3f} | {tr} | {er} | {sum(ct)} / {max(ct) if ct else 0} | {min(pt)}-{max(pt)} |")
    ref_key, ref = arms[0]
    for key, lab in arms[1:]:
        r0, r1 = data[ref], data[lab]
        names = sorted(set(r0) & set(r1), key=lambda n: man[n]["idx"])
        same_pt = sum(r0[n]["speed"]["prompt_tokens"] == r1[n]["speed"]["prompt_tokens"] for n in names)
        L.append(f"\n### Paired: {ref} (reference) vs {lab}, {len(names)} items scored in both arms (prompt token count equal in {same_pt})\n")

        def tab(sub):
            ab = [(bool(r0[n]["success"]), bool(r1[n]["success"])) for n in sub]
            A = sum(x and y for x, y in ab)
            B = sum(x and not y for x, y in ab)   # reference right, other wrong
            C = sum(y and not x for x, y in ab)   # other right, reference wrong
            D = sum((not x) and (not y) for x, y in ab)
            return A, B, C, D
        A, B, C, D = tab(names)
        n = len(names)
        diff, lo, hi = newcombe_paired(A, B, C, D) if n else (0, 0, 0)
        # newcombe_paired is p(first) - p(second) with first = reference; report other - reference
        L.append("| both right | only %s right | only %s right | both wrong | exact McNemar p (two-sided) | %s - %s accuracy | Newcombe 95%% CI of that difference |" % (ref, lab, lab, ref))
        L.append("|---|---|---|---|---|---|---|")
        L.append(f"| {A} | {B} | {C} | {D} | {mcnemar_exact(B, C):.3f} | {-diff:+.3f} ({-diff * n:+.1f} items) | {-hi:+.3f} to {-lo:+.3f} |")
        L.append(f"\nLargest loss of {lab} against {ref} that the 95% CI still allows: {max(0.0, hi) * 100:.1f} percentage points ({max(0.0, hi) * n:.1f} of {n} items).\n")
        # sensitivity: the string constants' expected values carry their double quotes; a reply without them has the right value
        def ok2(c):
            e = c["expected_output"]
            return " ".join(e.strip('"').casefold().split()) in " ".join((c["actual_output"] or "").casefold().split())
        s0 = [ok2(r0[n_]) for n_ in names]
        s1 = [ok2(r1[n_]) for n_ in names]
        A2 = sum(x and y for x, y in zip(s0, s1)); B2 = sum(x and not y for x, y in zip(s0, s1)); C2 = sum(y and not x for x, y in zip(s0, s1)); D2 = n - A2 - B2 - C2
        d2, lo2, hi2 = newcombe_paired(A2, B2, C2, D2)
        L.append(f"Sensitivity (quote-insensitive scoring of the string constants, same replies): {ref} {sum(s0)}/{n}, {lab} {sum(s1)}/{n}; only {ref} right {B2}, only {lab} right {C2}; McNemar p {mcnemar_exact(B2, C2):.3f}; {lab} - {ref} {-d2:+.3f}, Newcombe 95% CI {-hi2:+.3f} to {-lo2:+.3f}.\n")
        diffs = [n_ for n_ in names if (r0[n_]["actual_output"] or "").strip() != (r1[n_]["actual_output"] or "").strip()]
        L.append(f"Items whose reply text differs between the arms (after strip): {len(diffs)} of {n}" + (": " + ", ".join(diffs) if diffs else "") + ".\n")
        L.append("Discordant items:\n")
        for n_ in names:
            x, y = bool(r0[n_]["success"]), bool(r1[n_]["success"])
            if x != y:
                L.append(f"- `{n_}` expected `{r0[n_]['expected_output']}`; {ref}: `{(r0[n_]['actual_output'] or '').strip()[:80]}` ({'right' if x else 'wrong'}); {lab}: `{(r1[n_]['actual_output'] or '').strip()[:80]}` ({'right' if y else 'wrong'})")
        L.append("\n**By kind**\n")
        L.append("| kind | items | %s correct | %s correct | only %s | only %s | both wrong |" % (ref, lab, ref, lab))
        L.append("|---|---|---|---|---|---|---|")
        for kd in ("needle", "needle_dist", "needle_hop", "qa_const"):
            sub = [n_ for n_ in names if kind_of(n_) == kd]
            if sub:
                A_, B_, C_, D_ = tab(sub)
                L.append(f"| {kd} | {len(sub)} | {A_ + B_} | {A_ + C_} | {B_} | {C_} | {D_} |")
        L.append("\n**By depth of the answer-bearing text** (fraction of the document before it; qa_const: before the asked file)\n")
        L.append("| depth | items | %s correct | %s correct | only %s | only %s | both wrong |" % (ref, lab, ref, lab))
        L.append("|---|---|---|---|---|---|---|")
        for nm, lo_, hi_ in (("early (< 0.33)", 0, 0.33), ("middle (0.33 to 0.67)", 0.33, 0.67), ("late (>= 0.67)", 0.67, 1.01)):
            sub = [n_ for n_ in names if lo_ <= man[n_]["depth"] < hi_]
            if sub:
                A_, B_, C_, D_ = tab(sub)
                L.append(f"| {nm} | {len(sub)} | {A_ + B_} | {A_ + C_} | {B_} | {C_} | {D_} |")
        L.append("\n**By kind and depth, reference arm only** (correct / items)\n")
        L.append("| kind | early | middle | late |\n|---|---|---|---|")
        for kd in ("needle", "needle_dist", "needle_hop", "qa_const"):
            row = []
            for lo_, hi_ in ((0, 0.33), (0.33, 0.67), (0.67, 1.01)):
                sub = [n_ for n_ in names if kind_of(n_) == kd and lo_ <= man[n_]["depth"] < hi_]
                row.append(f"{sum(bool(r0[n_]['success']) for n_ in sub)}/{len(sub)}")
            L.append(f"| {kd} | " + " | ".join(row) + " |")
    open(f"{a.out}/analysis.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
