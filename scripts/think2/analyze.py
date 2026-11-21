#!/usr/bin/env python3
"""think2: tables from localeval thinking-on result.json files (greedy, same question ids on every arm).
usage: analyze.py --tok tokenizer.json [--json OUT] [--examples N] label=path/to/result.json ...   (first benchmark of each file; run with the localeval venv: cd /opt/code/deepeval && uv run python <this> ...)
Per run: score and 2 SE (binomial, sqrt(p(1-p)/n)), truncated, finish reasons, thinking / answer / total tokens (the checkpoint tokenizer
on reasoning_content and on the reply; the sum is checked against the server's completion_tokens), reply format checks.
Per pair: agreement on correctness, only-first / only-second right, sign test on total tokens, ratio of means, and disagreement examples."""
import argparse, json, math, random, statistics as st, itertools, re

def load(p):
    d = json.load(open(p)); b = d["benchmarks"][0]
    return {r["id"]: r for r in b["replies"]}, b, d

def ok(r):
    # gsm8k: localeval's primary metric is `flexible` (the record's `score` is the bare-number one); other benchmarks: `score`
    if r.get("flexible") is not None:
        return 1 if r["flexible"] else 0
    return 1 if (r.get("score") or 0) >= 1 else 0

def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]

def binom_two_sided(k, n):
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)

def bootstrap_ratio(a, b, n=4000, seed=1):
    rnd = random.Random(seed); idx = range(len(a)); rs = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        rs.append(sum(a[i] for i in s) / max(1, sum(b[i] for i in s)))
    rs.sort(); return rs[int(.025 * n)], rs[int(.975 * n)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tok"); ap.add_argument("--json"); ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("runs", nargs="+"); a = ap.parse_args()
    tk = None
    if a.tok:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tok)
    ntok = (lambda s: len(tk.encode(s, add_special_tokens=False).ids) if s else 0) if tk else (lambda s: round(len(s or "") / 4))
    runs = {}
    for x in a.runs:
        k, p = x.split("=", 1); runs[k] = load(p)
    out = {"runs": {}, "pairs": {}}
    print("| run | n | score | 2 SE | truncated | finish stop / length / other | thinking tok mean (median, p90, max) | answer tok mean | total tok mean | empty / no-thinking / inline `<think>` | wall s | tokenizer sum vs server |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    per = {}
    for k, (rep, b, d) in runs.items():
        n = len(rep); s = sum(ok(r) for r in rep.values()); pr = s / n; se2 = 2 * math.sqrt(pr * (1 - pr) / n)
        th = {i: ntok(r.get("reasoning_content")) for i, r in rep.items()}; an = {i: ntok(r.get("reply")) for i, r in rep.items()}
        tot = {i: r.get("completion_tokens") or 0 for i, r in rep.items()}
        per[k] = {"th": th, "an": an, "tot": tot}
        fr = [r.get("finish_reason") for r in rep.values()]
        empty = sum(1 for r in rep.values() if not (r.get("reply") or "").strip())
        nothink = sum(1 for r in rep.values() if not (r.get("reasoning_content") or "").strip())
        inline = sum(1 for r in rep.values() if "<think>" in (r.get("reply") or "") or "</think>" in (r.get("reply") or ""))
        diff = [abs(th[i] + an[i] - tot[i]) for i in rep]
        tv = list(th.values())
        row = {"n": n, "correct": s, "score": round(pr, 4), "two_se": round(se2, 4), "truncated": sum(1 for r in rep.values() if r.get("truncated")),
               "finish": {x: fr.count(x) for x in set(fr)}, "think_mean": round(st.mean(tv), 1), "think_median": st.median(tv), "think_p90": pct(tv, .9), "think_max": max(tv),
               "answer_mean": round(st.mean(an.values()), 1), "total_mean": round(st.mean(tot.values()), 1), "empty": empty, "no_thinking": nothink, "inline_think": inline,
               "wall_s": b.get("wall_s"), "tok_sum_vs_server_mean_abs": round(st.mean(diff), 1), "mean_output_tokens_localeval": b.get("mean_output_tokens"),
               "scores": b.get("scores")}
        if b.get("task_scores"):
            row["task_scores"] = b["task_scores"]
        out["runs"][k] = row
        print(f"| {k} | {n} | {pr:.3f} ({s}) | {se2:.3f} | {row['truncated']} | {fr.count('stop')} / {fr.count('length')} / {n - fr.count('stop') - fr.count('length')} | "
              f"{row['think_mean']} ({row['think_median']}, {row['think_p90']}, {row['think_max']}) | {row['answer_mean']} | {row['total_mean']} | {empty} / {nothink} / {inline} | {row['wall_s']} | {row['tok_sum_vs_server_mean_abs']} |")
    print()
    num_only = re.compile(r'^\s*\$?-?[\d,]+(\.\d+)?\s*%?\.?\s*$')
    if any(i.startswith("gsm8k") for i in next(iter(runs.values()))[0]):
        print("| run | number-only final answers | worked final answers (anything else) | worked: right of count |")
        print("|---|---|---|---|")
    for k, (rep, b, d) in runs.items():
        if not any(i.startswith("gsm8k") for i in rep):
            continue
        no = [r for r in rep.values() if num_only.match(r.get("reply") or "")]; wk = [r for r in rep.values() if not num_only.match(r.get("reply") or "")]
        print(f"| {k} | {len(no)} | {len(wk)} | {sum(ok(r) for r in wk)} of {len(wk)} |")
    if any(":" in i for i in next(iter(runs.values()))[0]):
        tasks = sorted({i.split(":", 1)[1] for i in next(iter(runs.values()))[0]})
        print()
        print("| task | " + " | ".join(f"{k}: score / truncated / mean tok" for k in runs) + " |")
        print("|---|" + "---|" * len(runs))
        for t in tasks:
            cells = []
            for k, (rep, b, d) in runs.items():
                rs = [r for i, r in rep.items() if i.endswith(":" + t)]
                cells.append(f"{sum(ok(r) for r in rs)}/{len(rs)} / {sum(1 for r in rs if r.get('truncated'))} / {round(st.mean(r.get('completion_tokens') or 0 for r in rs))}")
            print(f"| {t} | " + " | ".join(cells) + " |")
    print()
    print("| pair | same correctness | only first right | only second right | both right | both wrong | McNemar p | total tok: ratio of means 2nd/1st (95 % bootstrap) | second longer on / shorter on (sign test p) | thinking tok ratio |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    exs = {}
    for x, y in itertools.combinations(runs, 2):
        rx, ry = runs[x][0], runs[y][0]; ids = sorted(set(rx) & set(ry))
        c = [(ok(rx[i]), ok(ry[i])) for i in ids]
        o1 = sum(1 for p, q in c if p and not q); o2 = sum(1 for p, q in c if q and not p)
        both = sum(1 for p, q in c if p and q); neither = sum(1 for p, q in c if not p and not q)
        tx = [per[x]["tot"][i] for i in ids]; ty = [per[y]["tot"][i] for i in ids]
        lo, hi = bootstrap_ratio(ty, tx)
        longer = sum(1 for p, q in zip(tx, ty) if q > p); shorter = sum(1 for p, q in zip(tx, ty) if q < p)
        thx = [per[x]["th"][i] for i in ids]; thy = [per[y]["th"][i] for i in ids]
        mp = binom_two_sided(min(o1, o2), o1 + o2) if o1 + o2 else 1.0
        sp = binom_two_sided(min(longer, shorter), longer + shorter) if longer + shorter else 1.0
        out["pairs"][f"{x}:{y}"] = {"n": len(ids), "same": o1 + o2 == 0 and len(ids) or len(ids) - o1 - o2, "only_first": o1, "only_second": o2, "both": both, "neither": neither,
                                    "mcnemar_p": round(mp, 4), "tok_ratio": round(sum(ty) / sum(tx), 3), "tok_ratio_ci": [round(lo, 3), round(hi, 3)], "longer": longer, "shorter": shorter, "sign_p": round(sp, 4),
                                    "think_ratio": round(sum(thy) / max(1, sum(thx)), 3)}
        print(f"| {x} vs {y} | {len(ids) - o1 - o2} of {len(ids)} | {o1} | {o2} | {both} | {neither} | {mp:.3f} | {sum(ty)/sum(tx):.3f} ({lo:.3f} to {hi:.3f}) | {longer} / {shorter} ({sp:.3f}) | {sum(thy)/max(1,sum(thx)):.3f} |")
        exs[(x, y)] = [i for i, (p, q) in zip(ids, c) if p != q]
    print()
    for (x, y), ids in exs.items():
        print(f"### disagreements {x} vs {y}: {len(ids)} ids")
        for i in ids[:a.examples]:
            rx, ry = runs[x][0][i], runs[y][0][i]
            print(f"- {i}: gold {rx.get('gold')!r}; {x}: extracted {(rx.get('extracted') if rx.get('extracted') is not None else rx.get('prediction'))!r} score {ok(rx)} tokens {per[x]['tot'][i]} (thinking {per[x]['th'][i]}); {y}: extracted {(ry.get('extracted') if ry.get('extracted') is not None else ry.get('prediction'))!r} score {ok(ry)} tokens {per[y]['tot'][i]} (thinking {per[y]['th'][i]})")
    for x, y in itertools.combinations(runs, 2):
        rx, ry = runs[x][0], runs[y][0]; ids = sorted(set(rx) & set(ry))
        big = sorted(ids, key=lambda i: -abs(per[y]["tot"][i] - per[x]["tot"][i]))[:5]
        print(f"### largest length differences {x} vs {y}")
        for i in big:
            print(f"- {i}: {x} {per[x]['tot'][i]} tok (score {ok(rx[i])}), {y} {per[y]['tot'][i]} tok (score {ok(ry[i])})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
