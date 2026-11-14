#!/usr/bin/env python3
"""soak2 tables. usage: table.py <raw-dir> soak|probe|spec|vs
  soak : per soak replay (S<seed>): requests (greedy / sampled), errors, TTFT sum / p50 / p90, decode tok/s mean and median per class,
         client ITL p99 median / max over requests (localeval's per-token-weighted figure: not for stall claims), agg output tok/s,
         cached frac, wall. Totals row = all replays pooled.
  probe: the K=1 probe replays P-greedy / P-sampled: decode tok/s per request and the spec counter diff (acceptance by class).
  spec : MTP counters per soak replay from the per-replay /metrics scrapes (serve-S.metrics.after-seedN; before = the previous scrape).
The sampler / store / gap table comes from scripts/soak/soak_table.py (same sample-S.tsv layout)."""
import glob, json, os, re, statistics, sys

raw, mode = sys.argv[1], sys.argv[2]


def rows_of(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def med(v):
    return statistics.median(v) if v else float("nan")


def mean(v):
    return statistics.mean(v) if v else float("nan")


def metrics(p):
    d = {}
    for l in open(p):
        if l.startswith("#") or not l.strip():
            continue
        k, _, v = l.rpartition(" ")
        try:
            d[k] = float(v)
        except ValueError:
            pass
    return d


def stats(rows, wall=None):
    ok = [x for x in rows if x["ok"]]
    tt = [x["ttft_s"] for x in ok if x["ttft_s"] is not None]
    dc = [x["decode_tok_s"] for x in ok if x["decode_tok_s"]]
    pt = sum(x["prompt_tokens"] or 0 for x in ok)
    ct = sum(x["cached_tokens"] or 0 for x in ok)
    p99 = [x["itl_ms_p99"] for x in ok if x.get("itl_ms_p99") is not None]
    return dict(n=len(rows), err=len(rows) - len(ok), ttft_sum=sum(tt), ttft_p50=med(tt), ttft_p90=q(tt, 0.9), dec_mean=mean(dc), dec_med=med(dc),
                p99_med=med(p99), p99_max=max(p99) if p99 else float("nan"), cached=ct / pt if pt else 0,
                out=sum(x["completion_tokens"] or 0 for x in ok), wall=wall or max([x["end_s"] for x in rows] or [1]))


def soak_files():  # only replays that finished (have a summary)
    fs = [p for p in glob.glob(f"{raw}/S[0-9]*-interleaved-seed*-k2.jsonl") if os.path.exists(p.replace(".jsonl", ".summary.json"))]
    return sorted(fs, key=lambda p: int(re.search(r"/S(\d+)-", p).group(1)))


if mode == "soak":
    print("| replay | requests (greedy+sampled) | errors | TTFT sum s | TTFT p50 s | TTFT p90 s | decode tok/s mean greedy / sampled | decode tok/s median greedy / sampled | client ITL p99 ms median / max | agg output tok/s | cached frac | wall s |")
    print("|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|")
    allr = []
    for p in soak_files():
        r = rows_of(p)
        allr += r
        s = stats(r)
        g = [x for x in r if not x.get("temperature")]
        t = [x for x in r if x.get("temperature")]
        sg, st = stats(g), stats(t)
        print(f"| {os.path.basename(p).split('-')[0]} | {s['n']} ({sg['n']}+{st['n']}) | {s['err']} | {s['ttft_sum']:.1f} | {s['ttft_p50']:.2f} | {s['ttft_p90']:.2f} | "
              f"{sg['dec_mean']:.1f} / {st['dec_mean']:.1f} | {sg['dec_med']:.1f} / {st['dec_med']:.1f} | {s['p99_med']:.0f} / {s['p99_max']:.0f} | {s['out']/s['wall']:.1f} | {s['cached']:.3f} | {s['wall']:.0f} |")
    g = [x for x in allr if not x.get("temperature")]
    t = [x for x in allr if x.get("temperature")]
    sg, st, sa = stats(g, 1), stats(t, 1), stats(allr, 1)
    print(f"\npooled: {sa['n']} requests, {sa['err']} errors; greedy {sg['n']} (decode mean {sg['dec_mean']:.1f}, median {sg['dec_med']:.1f} tok/s, TTFT p50 {sg['ttft_p50']:.2f} s, p90 {sg['ttft_p90']:.2f} s, "
          f"cached {sg['cached']:.3f}); sampled {st['n']} (decode mean {st['dec_mean']:.1f}, median {st['dec_med']:.1f} tok/s, TTFT p50 {st['ttft_p50']:.2f} s, p90 {st['ttft_p90']:.2f} s, cached {st['cached']:.3f}); "
          f"all: TTFT p50 {sa['ttft_p50']:.2f} s, p90 {sa['ttft_p90']:.2f} s, p99 {q([x['ttft_s'] for x in allr if x['ok'] and x['ttft_s'] is not None], .99):.2f} s, client ITL p99 median/max over requests {sa['p99_med']:.0f} / {sa['p99_max']:.0f} ms")
    for key, lab in (("completion_tokens", "completion tokens"),):
        print(f"mean {lab}: greedy {mean([x[key] for x in g if x['ok']]):.1f}, sampled {mean([x[key] for x in t if x['ok']]):.1f}; finish_reason length share: greedy "
              f"{mean([x['finish_reason'] == 'length' for x in g if x['ok']]):.3f}, sampled {mean([x['finish_reason'] == 'length' for x in t if x['ok']]):.3f}")
elif mode == "probe":
    print("| probe | requests | errors | decode tok/s mean / median | TTFT sum s | spec steps | proposed | accepted | acceptance | tokens per spec step (1 + accepted/steps) |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for cls in ("greedy", "sampled"):
        fs = glob.glob(f"{raw}/P-{cls}-interleaved-seed*-k1.jsonl")
        if not fs:
            continue
        r = rows_of(fs[0])
        s = stats(r)
        b, a = metrics(f"{raw}/serve-S.metrics.before-P-{cls}"), metrics(f"{raw}/serve-S.metrics.after-P-{cls}")
        d = lambda k: a.get(k, 0) - b.get(k, 0)  # noqa: E731
        st, pr, ac = d("intarsia_spec_steps_total"), d("intarsia_spec_proposed_tokens_total"), d("intarsia_spec_accepted_tokens_total")
        print(f"| {cls} (T={'1.0' if cls == 'sampled' else '0'}) | {s['n']} | {s['err']} | {s['dec_mean']:.1f} / {s['dec_med']:.1f} | {s['ttft_sum']:.1f} | {st:.0f} | {pr:.0f} | {ac:.0f} | {ac/pr if pr else float('nan'):.3f} | {1 + ac/st if st else float('nan'):.2f} |")
elif mode == "spec":
    print("| replay | spec steps | proposed | accepted | acceptance | generated tokens | tokens in spec steps (steps + accepted) / generated |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    prev = metrics(f"{raw}/serve-S.metrics.after-P-sampled") if os.path.exists(f"{raw}/serve-S.metrics.after-P-sampled") else metrics(f"{raw}/serve-S.metrics.before")
    first = prev
    tot = [0, 0, 0, 0]
    for p in soak_files():
        seed = re.search(r"/S(\d+)-", p).group(1)
        f = f"{raw}/serve-S.metrics.after-seed{seed}"
        if not os.path.exists(f):
            continue
        cur = metrics(f)
        d = lambda k: cur.get(k, 0) - prev.get(k, 0)  # noqa: E731
        st, pr, ac, gen = d("intarsia_spec_steps_total"), d("intarsia_spec_proposed_tokens_total"), d("intarsia_spec_accepted_tokens_total"), d("intarsia_tokens_generated_total")
        tot = [tot[0] + st, tot[1] + pr, tot[2] + ac, tot[3] + gen]
        print(f"| S{seed} | {st:.0f} | {pr:.0f} | {ac:.0f} | {ac/pr if pr else float('nan'):.3f} | {gen:.0f} | {(st+ac)/gen if gen else float('nan'):.3f} |")
        prev = cur
    print(f"| total | {tot[0]:.0f} | {tot[1]:.0f} | {tot[2]:.0f} | {tot[2]/tot[1] if tot[1] else float('nan'):.3f} | {tot[3]:.0f} | {(tot[0]+tot[2])/tot[3] if tot[3] else float('nan'):.3f} |")
