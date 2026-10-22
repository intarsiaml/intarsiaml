#!/usr/bin/env python3
"""fx3: aggregate the interleaved Spark visits.

usage: [VISITS=<regex on visit id, e.g. '^p' or '^r'>] [REF=intarsia] aggregate.py <raw/<model> dir> [--out table.md]
Visit ids: pN = vLLM up (production server resident), rN = vLLM-down window, p9N = quality.

Files: <arm>-<pass>-speed.result.json, -conc.result.json, -quality.result.json (localeval results, one visit = one server start).
Per arm and prompt size, each visit contributes the median of its reps; a cell is the median of the visit medians, the band is
min..max over visits, n = number of visits. Two arms are TIED where their bands overlap. Decode and prefill ratios are against the
arm named REF (default intarsia).
"""
import glob, json, os, re, statistics as st, sys

ORDER = ["intarsia", "intarsia4k", "vllm", "sglang", "llamacpp", "llamacpp16"]
SIZES = [128, 1024, 4096, 16384]


def load(d, kind):
    out = {}
    vis = os.environ.get('VISITS')
    for f in sorted(glob.glob(os.path.join(d, f"*-{kind}.result.json"))):
        m = re.match(r"(.+?)-(p\d+|r\d+|smoke\d*)-" + kind + r"\.result\.json", os.path.basename(f))
        if not m or m.group(2).startswith("smoke"):
            continue
        arm, visit = m.groups()[:2]
        if vis and kind != 'quality' and not re.search(vis, visit):
            continue
        try:
            out.setdefault(arm, {})[visit] = json.load(open(f))
        except Exception:
            pass
    return out


def row(res, size):
    for r in (res.get("speed") or {}).get("rows", []):
        if r["target_prompt_tokens"] == size:
            return r
    return None


def cell(visits, size, key, sub="median"):
    vals = []
    for res in visits.values():
        r = row(res, size)
        if not r:
            continue
        v = r.get(key)
        if isinstance(v, dict):
            v = v.get(sub)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return {"med": st.median(vals), "lo": min(vals), "hi": max(vals), "n": len(vals), "vals": vals}


def fmt(c, nd=0):
    if c is None:
        return "-"
    f = f"{{:.{nd}f}}"
    return f"{f.format(c['med'])} ({f.format(c['lo'])}-{f.format(c['hi'])})"


def overlap(a, b):
    return a and b and not (a["hi"] < b["lo"] or b["hi"] < a["lo"])


def ratio_txt(c, ref):
    if not c or not ref:
        return "-"
    r = c["med"] / ref["med"]
    tag = "tie" if overlap(c, ref) else ("ahead" if r > 1 else "behind")
    return f"{r:.2f}x {tag}"


def main():
    d = sys.argv[1]
    ref = os.environ.get("REF", "intarsia")
    spd, conc, qual = load(d, "speed"), load(d, "conc"), load(d, "quality")
    arms = [a for a in ORDER if a in spd] + [a for a in spd if a not in ORDER]
    out = []
    out.append("### Single stream (concurrency 1): medians of visit medians, (min-max over visits)\n")
    out.append("| engine | n visits | " + " | ".join(f"decode {s}" for s in SIZES) + " |")
    out.append("|---|---:|" + "---|" * len(SIZES))
    for a in arms:
        out.append(f"| {a} | {len(spd[a])} | " + " | ".join(fmt(cell(spd[a], s, 'decode_tok_s'), 1) for s in SIZES) + " |")
    out.append("")
    out.append("| engine | " + " | ".join(f"prefill {s} (per-request)" for s in SIZES[1:]) + " | marginal 1k-4k / 4k-16k | TTFT 1k / 4k / 16k (s) |")
    out.append("|---|" + "---|" * (len(SIZES) - 1) + "---|---|")
    for a in arms:
        marg = []
        for s in (4096, 16384):
            marg.append(fmt(cell(spd[a], s, "marginal_prefill_tok_s", sub=None) if False else cell(spd[a], s, "marginal_prefill_tok_s"), 0))
        ttft = [fmt(cell(spd[a], s, "ttft_s"), 2) for s in SIZES[1:]]
        out.append(f"| {a} | " + " | ".join(fmt(cell(spd[a], s, "prefill_tok_s"), 0) for s in SIZES[1:]) + f" | {' / '.join(marg)} | {' / '.join(ttft)} |")
    out.append("")
    if ref in spd:
        out.append(f"### Ratio to {ref} (median of visit medians; tie = bands overlap)\n")
        out.append("| engine | " + " | ".join(f"decode {s}" for s in SIZES) + " | " + " | ".join(f"prefill {s}" for s in SIZES[1:]) + " |")
        out.append("|---|" + "---|" * (len(SIZES) + len(SIZES) - 1))
        for a in arms:
            if a == ref:
                continue
            dec = [ratio_txt(cell(spd[a], s, "decode_tok_s"), cell(spd[ref], s, "decode_tok_s")) for s in SIZES]
            pre = [ratio_txt(cell(spd[a], s, "prefill_tok_s"), cell(spd[ref], s, "prefill_tok_s")) for s in SIZES[1:]]
            out.append(f"| {a} | " + " | ".join(dec + pre) + " |")
        out.append("")
    # noise band: spread of visit medians as % of the median, per engine, pooled over sizes
    out.append("### Run-to-run noise (visit medians): (max-min)/median in %\n")
    out.append("| engine | " + " | ".join(f"decode {s}" for s in SIZES) + " | " + " | ".join(f"prefill {s}" for s in SIZES[1:]) + " |")
    out.append("|---|" + "---|" * (len(SIZES) + len(SIZES) - 1))
    for a in arms:
        cs = [cell(spd[a], s, "decode_tok_s") for s in SIZES] + [cell(spd[a], s, "prefill_tok_s") for s in SIZES[1:]]
        out.append(f"| {a} | " + " | ".join(("-" if c is None or c["n"] < 2 else f"{(c['hi'] - c['lo']) / c['med'] * 100:.1f}") for c in cs) + " |")
    out.append("")
    if conc:
        out.append("### Concurrency (1k-token prompts, 128 generated tokens, one burst per visit; median over visits)\n")
        streams = sorted({c["streams"] for v in conc.values() for res in v.values() for c in ((res.get("speed") or {}).get("concurrency") or [])})
        out.append("| engine | n | " + " | ".join(f"{s}: agg tok/s / per-stream / TTFT s / errors" for s in streams) + " |")
        out.append("|---|---:|" + "---|" * len(streams))
        for a in [x for x in ORDER if x in conc] + [x for x in conc if x not in ORDER]:
            cells = []
            for s in streams:
                agg, per, tt, er = [], [], [], 0
                for res in conc[a].values():
                    for c in ((res.get("speed") or {}).get("concurrency") or []):
                        if c["streams"] == s:
                            if c["aggregate_decode_tok_s"] is not None:
                                agg.append(c["aggregate_decode_tok_s"])
                            if c["per_stream_decode_tok_s"]:
                                per.append(c["per_stream_decode_tok_s"]["median"])
                            if c["ttft_s"]:
                                tt.append(c["ttft_s"]["median"])
                            er += len(c["errors"])
                cells.append("-" if not agg else f"{st.median(agg):.0f} ({min(agg):.0f}-{max(agg):.0f}) / {st.median(per):.1f} / {st.median(tt):.2f} / {er}")
            out.append(f"| {a} | {len(conc[a])} | " + " | ".join(cells) + " |")
        out.append("")
    if qual:
        out.append("### Quality (localeval quality tier, same bench/limit/structured for every engine)\n")
        out.append("| engine | visits | benchmark | score | n predictions |")
        out.append("|---|---:|---|---|---:|")
        for a in [x for x in ORDER if x in qual] + [x for x in qual if x not in ORDER]:
            for res in list(qual[a].values())[:1]:
                for b in res.get("benchmarks") or []:
                    out.append(f"| {a} | {len(qual[a])} | {b['benchmark']} | {b.get('overall_score')} | {b.get('n_predictions')} |")
        out.append("")
    txt = "\n".join(out)
    if "--out" in sys.argv:
        open(sys.argv[sys.argv.index("--out") + 1], "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
