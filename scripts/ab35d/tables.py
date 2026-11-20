#!/usr/bin/env python3
"""ab35d: markdown tables from the raw dir of the cutover scoreboard (stdlib only).
usage: tables.py RAW_DIR  > tables.md
Reads: <arm>-interleaved-seed<S>-k2.{jsonl,summary.json} (replay), snap/*.gz (per-cell /metrics of both engines),
       <arm>-sampled.json, <arm>-tools*.json, localeval result.json dirs listed in index.txt, marks-*.tsv + window-*.log (clock guard)."""
import gzip, json, pathlib, re, statistics, sys

RAW = pathlib.Path(sys.argv[1])


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def f(x, n=1):
    return "-" if x is None else f"{x:.{n}f}"


def hist(path, name):
    out = {}
    try:
        txt = gzip.open(path, "rt").read()
    except Exception:
        return out
    for line in txt.splitlines():
        if line.startswith(name + "_bucket"):
            le = line.split('le="')[1].split('"')[0]
            out[le] = out.get(le, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def counter(path, name):
    try:
        txt = gzip.open(path, "rt").read()
    except Exception:
        return None
    tot = 0.0
    seen = False
    for line in txt.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            tot += float(line.rsplit(" ", 1)[1]); seen = True
    return tot if seen else None


def gaps(cell, side):
    """server-side inter-token gap counts for a replay cell: (total, >0.15, >0.3 or >0.25, >0.5)."""
    name = "vllm:inter_token_latency_seconds" if side == "vllm" else "intarsia_batch_itl_seconds"
    a = hist(RAW / "snap" / f"{cell}.{side}.after.gz", name)
    b = hist(RAW / "snap" / f"{cell}.{side}.before.gz", name)
    if not a:
        return None
    d = {k: a[k] - b.get(k, 0.0) for k in a}
    n = d.get("+Inf", 0)
    mid = "0.3" if "0.3" in d else "0.25"
    return int(n), int(n - d.get("0.15", 0)), (mid, int(n - d.get(mid, 0))), int(n - d.get("0.5", 0))


print("## Replay (2 streams, interleaved, 48 requests)\n")
print("| arm | seed | errors | TTFT sum s | TTFT p50 / p90 s | cached frac | decode tok/s per stream (median) | combined out tok/s | wall s | client ITL p99 ms median / max | server gaps >0.15 / >0.3 (vLLM) or >0.25 (B) / >0.5 of N |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for p in sorted(RAW.glob("[AB]-interleaved-seed*-k2.jsonl")):
    arm, seed = re.match(r"(\w+)-interleaved-seed(\d+)-k2", p.name).groups()
    rows = [json.loads(l) for l in open(p)]
    ok = [r for r in rows if r["ok"]]
    summ = json.load(open(p.with_suffix(".summary.json")))
    tt = sorted(r["ttft_s"] for r in ok)
    toks = sum(r["completion_tokens"] for r in ok)
    pt = sum(r["prompt_tokens"] for r in ok)
    ct = sum(r["cached_tokens"] for r in ok)
    p99 = [r["itl_ms_p99"] for r in ok if r.get("itl_ms_p99") is not None]
    side = "vllm" if arm == "A" else "b"
    g = gaps(f"replay-{arm}-s{seed}", side)
    gs = "-" if not g else f"{g[1]} / {g[2][1]} / {g[3]} of {g[0]}"
    print(f"| {arm} | {seed} | {len(rows) - len(ok)} | {sum(tt):.1f} | {tt[len(tt) // 2]:.2f} / {tt[int(len(tt) * 0.9)]:.2f} | {ct / pt:.3f} | {f(med([r['decode_tok_s'] for r in ok]))} | {toks / summ['wall_s']:.1f} | {summ['wall_s']:.1f} | {f(med(p99), 0)} / {f(max(p99) if p99 else None, 0)} | {gs} |")

print("\n## Per-turn TTFT median s (cached tokens median), replay\n")
turns = {}
for p in sorted(RAW.glob("[AB]-interleaved-seed*-k2.jsonl")):
    arm, seed = re.match(r"(\w+)-interleaved-seed(\d+)-k2", p.name).groups()
    for r in (json.loads(l) for l in open(p)):
        if r["ok"]:
            turns.setdefault((arm, seed), {}).setdefault(r["turn"], []).append(r)
keys = sorted(turns)
print("| turn | " + " | ".join(f"{a} s{s}" for a, s in keys) + " |")
print("|---|" + "---|" * len(keys))
for t in range(1, 7):
    cells = []
    for k in keys:
        rs = turns[k].get(t, [])
        cells.append(f"{med([r['ttft_s'] for r in rs]):.2f} ({int(med([r['cached_tokens'] for r in rs]))})" if rs else "-")
    print(f"| {t} | " + " | ".join(cells) + " |")

print("\n## B MTP counters per replay (diff of /metrics)\n")
print("| cell | spec steps | proposed | accepted | acceptance | completion tokens | share of tokens made in spec steps (steps+accepted)/tokens |")
print("|---|---|---|---|---|---|---|")
for seed in ("0", "1", "2", "3"):
    cell = f"replay-B-s{seed}"
    d = {}
    for n in ("steps", "proposed_tokens", "accepted_tokens"):
        a = counter(RAW / "snap" / f"{cell}.b.after.gz", f"intarsia_spec_{n}_total")
        b = counter(RAW / "snap" / f"{cell}.b.before.gz", f"intarsia_spec_{n}_total")
        d[n] = None if a is None or b is None else a - b
    jp = RAW / f"B-interleaved-seed{seed}-k2.jsonl"
    if not jp.exists() or d["steps"] is None:
        continue
    toks = sum(r["completion_tokens"] for r in map(json.loads, open(jp)) if r["ok"])
    acc = d["accepted_tokens"] / d["proposed_tokens"] if d["proposed_tokens"] else None
    print(f"| {cell} | {d['steps']:.0f} | {d['proposed_tokens']:.0f} | {d['accepted_tokens']:.0f} | {f(acc and acc * 100)} % | {toks} | {f((d['steps'] + d['accepted_tokens']) / toks, 3)} |")

sj = {p.name.split("-sampled")[0]: json.load(open(p)) for p in sorted(RAW.glob("*-sampled.json"))}


def sampled_table(title, order, a_pref, b_pref):
    order = [k for k in order if k in sj]
    if not order:
        return
    print(f"\n## {title}\n")
    sizes = sorted({r["size"] for k in order for r in sj[k]["rows"] if "decode_tok_s" in r})
    for metric, key, nd in (("decode tok/s", "decode_tok_s", 1), ("TTFT s", "ttft_s", 2)):
        print(f"\n{metric}: median of the 3 reps per run (reps), runs in ABBA order\n")
        print("| size (real prompt tokens) | " + " | ".join(order) + " | A mean | arm mean | arm / A |")
        print("|---|" + "---|" * (len(order) + 3))
        for sz in sizes:
            vals, cells, pr = {}, [], None
            for k in order:
                r = next((x for x in sj[k]["rows"] if x["size"] == sz and "decode_tok_s" in x), None)
                if not r:
                    cells.append("-"); continue
                pr = r["prompt_tokens"]
                vals[k] = r[key]["median"]
                reps = ", ".join(f"{x:.{nd}f}" for x in r["decode_each"]) if key == "decode_tok_s" else f"{r[key]['min']:.{nd}f}-{r[key]['max']:.{nd}f}"
                cells.append(f"{r[key]['median']:.{nd}f} ({reps})")
            am = statistics.mean(v for k, v in vals.items() if k.startswith(a_pref))
            bm = statistics.mean(v for k, v in vals.items() if k.startswith(b_pref))
            print(f"| {sz} ({pr}) | " + " | ".join(cells) + f" | {am:.{nd}f} | {bm:.{nd}f} | {bm / am:.3f} |")
    print("\nB MTP acceptance on the sampled requests (accepted / proposed from the /metrics diff over the run's 3 reps):\n")
    for k in order:
        if k.startswith(b_pref):
            print(f"- {k}: " + ", ".join(f"{r['size']}: {r['mtp']['accept_rate'] * 100:.1f} % ({r['mtp']['tokens_per_verify']} tokens per verify)" for r in sj[k]["rows"] if r.get("mtp") and r["mtp"].get("accept_rate") is not None))


sampled_table("Sampled decode, shipping config (T=0.7, top_p 0.95, 256 forced tokens, 3 reps, nonce prompts)", ["A1", "B1", "B2", "A2"], "A", "B")
sampled_table("Control: sampled decode with `--kv-dtype fp8` on B (window w8; A is the same vLLM)", ["ctl-A1", "ctl-N1", "ctl-N2", "ctl-A2"], "ctl-A", "ctl-N")

def index():
    out = {}
    p = RAW / "index.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.match(r"(\S+) run_id=(\S+)", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


IDX = index()


def speed_tables(title, keys, a_names, b_names):
    sp = {k: json.load(open(RAW / IDX[k] / "result.json")) for k in keys if k in IDX and (RAW / IDX[k] / "result.json").exists()}
    if not sp:
        return
    print(f"\n## {title}\n")
    order = [k for k in keys if k in sp]
    sizes = [r["target_prompt_tokens"] for r in sp[order[0]]["speed"]["rows"]]
    for metric, key, nd in (("decode tok/s", "decode_tok_s", 1), ("TTFT s", "ttft_s", 3), ("prefill tok/s (uncached prompt tokens / TTFT)", "prefill_tok_s", 0)):
        print(f"\n{metric}: median of the 3 reps per run (min-max), runs in ABBA order\n")
        print("| size (real prompt tokens) | " + " | ".join(order) + " | A mean | arm mean | arm / A |")
        print("|---|" + "---|" * (len(order) + 3))
        for i, sz in enumerate(sizes):
            vals, cells = {}, []
            for k in order:
                r = sp[k]["speed"]["rows"][i][key]
                vals[k] = r["median"]
                cells.append(f"{r['median']:.{nd}f} ({r['min']:.{nd}f}-{r['max']:.{nd}f})")
            am = statistics.mean(v for k, v in vals.items() if any(n in k for n in a_names))
            bm = statistics.mean(v for k, v in vals.items() if any(n in k for n in b_names))
            pr = sp[order[0]]["speed"]["rows"][i]["prompt_tokens"]
            print(f"| {sz} ({pr}) | " + " | ".join(cells) + f" | {am:.{nd}f} | {bm:.{nd}f} | {bm / am:.3f} |")


speed_tables("Greedy single-stream speed, shipping config B (store on) vs A (localeval speed, nonce prompts, 256 forced tokens, 3 reps + 1 warmup per run)",
             ["A1-speed", "B1-speed", "B2-speed", "A2-speed"], ["A1", "A2"], ["B1", "B2"])
speed_tables("Control: same, arm N = `--max-batch 2` without the prefix store (window w5)",
             ["ctl-A1-speed", "ctl-N1-speed", "ctl-N2-speed", "ctl-A2-speed"], ["ctl-A"], ["ctl-N"])

print("\n## Clock guard and shared-box check per cell (from window-*.log)\n")
print("| window | cell | samples | at util>90 | of those SM<2000 | min SM MHz | mean SM MHz | max W | SW power cap delta us | vLLM delta (requests / gen tokens) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for lp in sorted(RAW.glob("window-*.log")):
    for line in lp.read_text().splitlines():
        m = re.search(r"cell (\S+) end: clock guard: (\d+) samples, (\d+) at util>90, (\d+) of those SM<2000 MHz, min SM (\d+), mean SM (\d+), max power (\d+) W; SW power cap delta (-?\d+) us; vLLM delta success (-?\d+) gen_tokens (-?\d+)", line)
        if m:
            c, n, u, lo, mn, mean, pw, cap, vs, vg = m.groups()
            print(f"| {lp.stem.split('-')[1]} | {c} | {n} | {u} | {lo} | {mn} | {mean} | {pw} | {cap} | {vs} / {vg} |")

print("\n## Replay, 1 stream, interleaved (blocker 3's original shape; window w7, B = store on)\n")
print("| arm | seed | errors | TTFT sum s | TTFT p50 / p90 s | cached frac | decode tok/s (median, mean) | combined out tok/s | wall s | server gaps >0.15 / >0.3 (vLLM) or >0.25 (B) / >0.5 of N |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in sorted(RAW.glob("ctl1-*-interleaved-seed*-k1.jsonl")):
    arm, seed = re.match(r"ctl1-(\w+)-interleaved-seed(\d+)-k1", p.name).groups()
    rows = [json.loads(l) for l in open(p)]
    ok = [r for r in rows if r["ok"]]
    summ = json.load(open(p.with_suffix(".summary.json")))
    tt = sorted(r["ttft_s"] for r in ok)
    toks = sum(r["completion_tokens"] for r in ok)
    pt = sum(r["prompt_tokens"] for r in ok)
    ct = sum(r["cached_tokens"] for r in ok)
    g = gaps(f"ctlreplay1-{arm}-s{seed}", "vllm" if arm == "A" else "b")
    gs = "-" if not g else f"{g[1]} / {g[2][1]} / {g[3]} of {g[0]}"
    dec = [r["decode_tok_s"] for r in ok]
    print(f"| {'B (raw name N)' if arm == 'N' else arm} | {seed} | {len(rows) - len(ok)} | {sum(tt):.1f} | {tt[len(tt) // 2]:.2f} / {tt[int(len(tt) * 0.9)]:.2f} | {ct / pt:.3f} | {f(med(dec))}, {f(statistics.mean(dec))} | {toks / summ['wall_s']:.1f} | {summ['wall_s']:.1f} | {gs} |")
