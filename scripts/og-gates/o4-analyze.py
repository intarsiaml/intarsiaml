#!/usr/bin/env python3
"""O4 analyzer: paired A/B of o4-ab.sh. Per prompt size: for every round the relative prefill difference
(dflt - off) / off in percent, its mean, the 95 % confidence interval of the mean (Student t, n-1 degrees of freedom) and the
sign count. RULE (fixed before the run): the 4096 delta is NOISE if the 95 % CI of the paired mean includes 0; it is a real but
small effect if the CI excludes 0 and |mean| < 1 %. Arms: dflt = defaults, off = INTARSIA_HIP_STAGE=0 + INTARSIA_HIP_SPLIT_MIN_US=0.

    o4-analyze.py DIR      # DIR holds o4-<arm>-r<k>.json from `intarsia bench speed`
"""
import glob, json, math, pathlib, re, statistics, sys

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
d = pathlib.Path(sys.argv[1])
rows = {}  # (arm, round) -> {size: row}
for f in sorted(glob.glob(str(d / "o4-*-r*.json"))):
    m = re.search(r"o4-(\w+)-r(\d+)\.json", f)
    j = json.loads(pathlib.Path(f).read_text())
    rows[(m.group(1), int(m.group(2)))] = {r["target_prompt_tokens"]: r for r in j["rows"]}
rounds = sorted({r for (_, r) in rows})
sizes = sorted({s for v in rows.values() for s in v})
for metric in ("prefill_tok_s", "decode_tok_s"):
    print(f"\n## {metric}")
    for s in sizes:
        print(f"\n### target {s}")
        print("| round | dflt | off | dflt tokens | off tokens | diff % |")
        print("|---|---|---|---|---|---|")
        diffs = []
        for r in rounds:
            a, b = rows.get(("dflt", r), {}).get(s), rows.get(("off", r), {}).get(s)
            if not a or not b:
                continue
            x = 100.0 * (a[metric] - b[metric]) / b[metric]
            diffs.append(x)
            print(f"| {r} | {a[metric]:.1f} | {b[metric]:.1f} | {a['prompt_tokens']:.0f} | {b['prompt_tokens']:.0f} | {x:+.2f} |")
        n = len(diffs)
        if n < 2:
            continue
        mean = statistics.mean(diffs); sd = statistics.stdev(diffs); half = T975.get(n - 1, 1.96) * sd / math.sqrt(n)
        pos = sum(1 for x in diffs if x > 0)
        verdict = "NOISE (CI includes 0)" if mean - half <= 0 <= mean + half else ("small real effect" if abs(mean) < 1 else "real effect")
        print(f"\nn={n}  mean {mean:+.2f} %  sd {sd:.2f}  95% CI [{mean - half:+.2f}, {mean + half:+.2f}]  positive in {pos}/{n}  -> {verdict}")
