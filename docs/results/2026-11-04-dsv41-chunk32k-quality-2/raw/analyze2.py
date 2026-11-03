#!/usr/bin/env python3
"""Paired analysis for set 2 (docs/results/2026-11-04-dsv41-chunk32k-quality-2.md).

Arms are localeval result.json labels dsv41-chunk32k2-<arm>-<shard>. The first arm is the reference (chunk 512); every other arm is
compared with it item by item (names are unique). For each comparison: discordant counts, exact McNemar, Newcombe paired 95% CI of
(arm - reference), the largest loss the CI allows, items whose final answer / reply text differ, by kind, level, size, span.
With three arms the "shift" of 2048 is also compared with the shift of the noise control (HC_FUSED=0): a sign test on the
items where exactly one of the two arms differs from the reference.

  python3 analyze2.py --arms ab0=512 ab1=2048 ab2=512-hc0 --since 20261103-2000 --manifest M.json --out DIR [--prefix dsv41-chunk32k2]
"""
import argparse, glob, json, math, os, re
from collections import defaultdict

FINAL = re.compile(r"FINAL:\s*\[?\s*([^\]\n]*?)\s*\]?\s*\.?\s*$", re.I | re.M)


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_paired(a, b, c, d, z=1.959964):
    """Newcombe (1998) method 10, CI of p1 - p2 (paired). a both, b only first, c only second, d neither."""
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


def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(results, prefix, arm, since):
    out = {}
    for f in sorted(glob.glob(f"{results}/*-adhoc-{prefix}-{arm}-*/result.json")):
        if os.path.basename(os.path.dirname(f)) < since:
            continue
        r = json.load(open(f))
        for d in r["datasets"]:
            for c in d["cases"]:
                out[c["name"]] = {**c, "run_id": r["run_id"]}
    return out


def final_of(c):
    m = FINAL.findall(c["actual_output"] or "")
    return m[-1].strip() if m else None


def exact(c):
    """The parsed FINAL line equals the expected answer exactly (stricter than localeval's substring check)."""
    f = final_of(c)
    return f is not None and f.casefold() == (c["expected_output"] or "")[len("FINAL:"):].strip().casefold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/opt/code/deepeval/results")
    ap.add_argument("--prefix", default="dsv41-chunk32k2")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--since", default="00000000")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = {m["name"]: m for m in json.load(open(a.manifest))["items"]}
    arms = [x.split("=") for x in a.arms]
    data = {lab: load(a.results, a.prefix, key, a.since) for key, lab in arms}
    for key, lab in arms:
        with open(f"{a.out}/replies-{key}.jsonl", "w") as f:
            for n in sorted(data[lab], key=lambda n: man[n]["idx"]):
                c = data[lab][n]
                sp = c["speed"]
                f.write(json.dumps({"name": n, "expected": c["expected_output"], "reply": c["actual_output"], "final": final_of(c), "success": bool(c["success"]),
                                    "prompt_tokens": sp["prompt_tokens"], "cached_tokens": sp["cached_tokens"], "completion_tokens": sp["completion_tokens"],
                                    "ttft_s": round(sp["ttft_s"], 2) if sp["ttft_s"] else None, "error": c["error"], "run_id": c["run_id"]}, ensure_ascii=False) + "\n")
    L = []
    L.append("### Accuracy per arm\n")
    L.append("| arm | items | correct | accuracy | Wilson 95%% CI | exact-parse correct | no FINAL line | truncated (>= %d tokens) | call errors | cached tokens (sum) | mean TTFT 32K / 65K (s) |" % a.max_tokens)
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, lab in arms:
        d = data[lab]
        k = sum(bool(c["success"]) for c in d.values())
        lo, hi = wilson(k, len(d))
        nofin = sum(1 for c in d.values() if final_of(c) is None)
        tr = sum(1 for c in d.values() if (c["speed"]["completion_tokens"] or 0) >= a.max_tokens)
        ct = sum((c["speed"]["cached_tokens"] or 0) for c in d.values())
        t = {32: [], 65: []}
        for n, c in d.items():
            t[man[n]["k"]].append(c["speed"]["ttft_s"] or 0)
        tt = " / ".join(f"{sum(v) / len(v):.1f}" if v else "-" for v in (t[32], t[65]))
        L.append(f"| {lab} ({key}) | {len(d)} | {k} | {k / max(len(d), 1):.3f} | {lo:.3f} to {hi:.3f} | {sum(exact(c) for c in d.values())} | {nofin} | {tr} | {sum(1 for c in d.values() if c['error'])} | {ct} | {tt} |")
    ref_key, ref = arms[0]
    r0 = data[ref]

    def tab(r1, sub):
        ab = [(bool(r0[n]["success"]), bool(r1[n]["success"])) for n in sub]
        A = sum(x and y for x, y in ab); B = sum(x and not y for x, y in ab); C = sum(y and not x for x, y in ab)
        return A, B, C, len(ab) - A - B - C

    for key, lab in arms[1:]:
        r1 = data[lab]
        names = sorted(set(r0) & set(r1), key=lambda n: man[n]["idx"])
        n = len(names)
        L.append(f"\n### Paired: {ref} (reference) vs {lab}: {n} items scored in both arms (equal prompt token count in {sum(r0[x]['speed']['prompt_tokens'] == r1[x]['speed']['prompt_tokens'] for x in names)})\n")
        A, B, C, D = tab(r1, names)
        diff, lo, hi = newcombe_paired(A, B, C, D) if n else (0, 0, 0)
        L.append(f"| both right | only {ref} right | only {lab} right | both wrong | exact McNemar p | {lab} - {ref} | Newcombe 95%% CI |".replace("%%", "%"))
        L.append("|---|---|---|---|---|---|---|")
        L.append(f"| {A} | {B} | {C} | {D} | {sign_p(B, C):.3f} | {-diff:+.3f} ({-diff * n:+.1f} items) | {-hi:+.3f} to {-lo:+.3f} |")
        L.append(f"\nLargest loss of {lab} against {ref} that the 95% CI allows: {max(0.0, hi) * 100:.1f} percentage points ({max(0.0, hi) * n:.1f} of {n} items).\n")
        txt = [x for x in names if (r0[x]["actual_output"] or "").strip() != (r1[x]["actual_output"] or "").strip()]
        fin = [x for x in names if final_of(r0[x]) != final_of(r1[x])]
        L.append(f"Reply text differs for {len(txt)} of {n} items; the final answer differs for {len(fin)} of {n}.\n")
        L.append("**By kind**\n")
        L.append(f"| kind | items | {ref} correct | {lab} correct | only {ref} | only {lab} | both wrong | final answer differs |\n|---|---|---|---|---|---|---|---|")
        for kd in ("hop2", "hop3", "agg_sum", "agg_max", "repo_sum"):
            sub = [x for x in names if man[x]["kind"] == kd]
            if sub:
                A_, B_, C_, D_ = tab(r1, sub)
                L.append(f"| {kd} | {len(sub)} | {A_ + B_} | {A_ + C_} | {B_} | {C_} | {D_} | {sum(1 for x in sub if final_of(r0[x]) != final_of(r1[x]))} |")
        for title, keyf, vals in (("level", lambda m: m["level"], (1, 2, 3)), ("size", lambda m: m["k"], (32, 65)),
                                  ("span of the facts (fraction of the document)", lambda m: 0 if m["span"] < 0.3 else 1 if m["span"] < 0.6 else 2, (0, 1, 2))):
            L.append(f"\n**By {title}**\n")
            L.append(f"| {title} | items | {ref} correct | {lab} correct | only {ref} | only {lab} | both wrong |\n|---|---|---|---|---|---|---|")
            for v in vals:
                sub = [x for x in names if keyf(man[x]) == v]
                if sub:
                    A_, B_, C_, D_ = tab(r1, sub)
                    nm = {0: "span < 0.3", 1: "0.3 to 0.6", 2: ">= 0.6"}[v] if title.startswith("span") else str(v)
                    L.append(f"| {nm} | {len(sub)} | {A_ + B_} | {A_ + C_} | {B_} | {C_} | {D_} |")
        L.append("\nDiscordant items: " + (", ".join(f"`{x}` ({ref} {'right' if r0[x]['success'] else 'wrong'}, {lab} {'right' if r1[x]['success'] else 'wrong'})" for x in names if bool(r0[x]['success']) != bool(r1[x]['success'])) or "none") + "\n")
    if len(arms) >= 3:
        (k1, l1), (k2, l2) = arms[1], arms[2]
        r1, r2 = data[l1], data[l2]
        names = sorted(set(r0) & set(r1) & set(r2), key=lambda n: man[n]["idx"])
        n = len(names)
        L.append(f"\n### Is the shift of {l1} larger than the noise control {l2}? ({n} items scored in all three arms)\n")
        d1 = {x: final_of(r0[x]) != final_of(r1[x]) for x in names}
        d2 = {x: final_of(r0[x]) != final_of(r2[x]) for x in names}
        t1 = {x: (r0[x]["actual_output"] or "").strip() != (r1[x]["actual_output"] or "").strip() for x in names}
        t2 = {x: (r0[x]["actual_output"] or "").strip() != (r2[x]["actual_output"] or "").strip() for x in names}
        o1 = {x: bool(r0[x]["success"]) != bool(r1[x]["success"]) for x in names}
        o2 = {x: bool(r0[x]["success"]) != bool(r2[x]["success"]) for x in names}
        L.append(f"| measure against {ref} | {l1} | {l2} | only {l1} moves | only {l2} moves | sign-test p (two-sided) |\n|---|---|---|---|---|---|")
        for nm, x1, x2 in (("reply text differs", t1, t2), ("final answer differs", d1, d2), ("correctness flips", o1, o2)):
            b = sum(1 for x in names if x1[x] and not x2[x]); c = sum(1 for x in names if x2[x] and not x1[x])
            L.append(f"| {nm} | {sum(x1.values())} | {sum(x2.values())} | {b} | {c} | {sign_p(b, c):.3f} |")
        acc = {lab: sum(bool(data[lab][x]["success"]) for x in names) for _, lab in arms}
        L.append("\nCorrect on these items: " + ", ".join(f"{lab} {v}/{n}" for lab, v in acc.items()) + ".")
        # 2048 versus HC0 directly (the two perturbations against each other)
        A, B, C, D = (sum(bool(r1[x]["success"]) and bool(r2[x]["success"]) for x in names), sum(bool(r1[x]["success"]) and not bool(r2[x]["success"]) for x in names),
                      sum(bool(r2[x]["success"]) and not bool(r1[x]["success"]) for x in names), 0)
        D = n - A - B - C
        dd, lo, hi = newcombe_paired(A, B, C, D)
        L.append(f"\n{l1} against {l2}: only {l1} right {B}, only {l2} right {C}, McNemar p {sign_p(B, C):.3f}; {l1} - {l2} {dd:+.3f}, Newcombe 95% CI {lo:+.3f} to {hi:+.3f}.\n")
    open(f"{a.out}/analysis.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
