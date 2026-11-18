#!/usr/bin/env python3
"""flipev: tables from raw/herd.jsonl and raw/q4.jsonl. usage: table.py <raw dir>"""
import json, sys, statistics as st, collections, glob

raw = sys.argv[1]
M = lambda v: sum(v) / len(v)  # noqa: E731


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def herd():
    rows = [json.loads(l) for l in open(raw + "/herd.jsonl")]
    conc = [r for r in rows if r["mode"] in ("herd8", "mix")]
    solo = [r for r in rows if r["mode"] == "solo"]
    ref = {}
    for r in solo:
        ref.setdefault((r["seed"], r.get("prompts", "herd")), r["ids"])
    def cell(r):
        if r["mode"] == "mix":
            return "mix 4 shared 16K + 2 unrelated 8K"
        return f"shared {r['args']['shared']} stagger {r['stagger']}"
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in conc:
        g[cell(r)][r["arm"]].append(r)
    print("## herd: TTFT sum / p50 / last / wall (s), mean over cells (range), n cells\n")
    print("| cell | arm | n | TTFT sum | TTFT p50 | TTFT last | wall | cached tokens (first, rest) |\n|---|---|---|---|---|---|---|---|")
    for c, arms in sorted(g.items()):
        for arm in "AB":
            rs = arms.get(arm, [])
            if not rs:
                continue
            f = lambda k: f"{M([r[k] for r in rs]):.2f} ({min(r[k] for r in rs):.2f} to {max(r[k] for r in rs):.2f})"  # noqa: E731
            cached = rs[0]["cached_tokens"]
            print(f"| {c} | {'off' if arm == 'A' else 'on'} | {len(rs)} | {f('ttft_sum_s')} | {f('ttft_p50_s')} | {f('ttft_last_s')} | {f('wall_s')} | {cached} |")
    print("\n## per request index (mean over cells): TTFT s / end-of-reply s since the barrier\n")
    for c, arms in sorted(g.items()):
        if "stagger 0.0" in c and "mix" not in c:
            continue
        print(f"### {c}\n\n| idx | send at | off TTFT | on TTFT | off end | on end |\n|---|---|---|---|---|---|")
        n = len(next(iter(arms.values()))[0]["ttft_s"])
        for i in range(n):
            vals = {}
            for arm in "AB":
                rs = arms.get(arm, [])
                vals[arm] = (M([r["ttft_s"][i] for r in rs]), M([r["abs_end_s"][i] for r in rs])) if rs else (float("nan"),) * 2
            send = M([r["send_s"][i] for r in arms["A"]]) if "A" in arms else float("nan")
            print(f"| {i} | {send:.2f} | {vals['A'][0]:.2f} | {vals['B'][0]:.2f} | {vals['A'][1]:.2f} | {vals['B'][1]:.2f} |")
        print()
    print("## herd ids: every reply (concurrent and solo) against the solo reply of the same prompt set (first visit)\n")
    tot = collections.Counter(); bad = collections.Counter(); errs = 0; cnts = collections.Counter()
    for r in rows:
        errs += sum(1 for e in r["errors"] if e)
        key = (r["seed"], "mix" if (r["mode"] == "mix" or r.get("prompts") == "mix") else "herd")
        for i, ids in enumerate(r["ids"]):
            k = cell(r) if r["mode"] != "solo" else None
            kk = key
            if kk not in ref:
                continue
            tot[(r["arm"], r["mode"])] += 1
            cnts[len(ids)] += 1
            if ids != ref[kk][i]:
                bad[(r["arm"], r["mode"])] += 1
    for k in sorted(tot):
        print(f"- arm {k[0]} {k[1]}: {tot[k] - bad[k]} / {tot[k]} equal")
    print(f"- errors: {errs}; reply lengths (tokens: count): {dict(cnts)}")


def q4():
    rows = [json.loads(l) for l in open(raw + "/q4.jsonl")]
    rows = [r for r in rows if "error" not in r]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["d"], r["scn"], r["arm"][0])].append(r)
    print("\n## q4: ITL of the ongoing decodes (event gaps, ms) while the 40K prompt prefills; pooled over runs\n")
    print("| decoders | scenario | arm | runs | window gaps n | p50 | p99 | max | base p50 | base p99 | base max | long TTFT s | short TTFTs s (s1 / s2) | dec tok/s/stream in window |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (d, scn, arm), rs in sorted(g.items()):
        gw = [x for r in rs for x in r["gaps_window"]]; gb = [x for r in rs for x in r["gaps_base"]]
        sh = [r["short_ttft"] for r in rs if r["short_ttft"]]
        s1 = f"{M([s['s1'] for s in sh]):.2f} / {M([s['s2'] for s in sh]):.2f}" if sh else "-"
        print(f"| {d} | {scn} | {'off' if arm == 'A' else 'on'} | {len(rs)} | {len(gw)} | {pct(gw, .5)*1e3:.1f} | {pct(gw, .99)*1e3:.1f} | {max(gw)*1e3:.1f} | {pct(gb, .5)*1e3:.1f} | {pct(gb, .99)*1e3:.1f} | {max(gb)*1e3:.1f} | {M([r['long_ttft'] for r in rs]):.2f} | {s1} | {M([r['dec_tok_s_in_window'] for r in rs]):.1f} |")
    print("\nper-run window max gap (ms) and p99:\n")
    for (d, scn, arm), rs in sorted(g.items()):
        print(f"- d{d} {scn} {'off' if arm == 'A' else 'on'}: max " + ", ".join(f"{r['itl_window']['max']*1e3:.0f}" for r in rs) + " | p99 " + ", ".join(f"{r['itl_window']['p99']*1e3:.0f}" for r in rs))
    print("\n## q4 ids\n")
    eq = collections.Counter(); tot = collections.Counter(); cnt = collections.Counter()
    for r in rows:
        for k, v in (r["ids_eq_solo"] or {}).items():
            tot[r["arm"][0]] += 1; eq[r["arm"][0]] += bool(v)
        for k, n in r["ids_n"].items():
            cnt[(k[0] if k[0] != "d" else "d", n)] += 1
    for a in "AB":
        print(f"- arm {'off' if a == 'A' else 'on'}: {eq[a]} / {tot[a]} contended replies equal their own server's solo reply")
    # across arms: same key, same ids
    byk = collections.defaultdict(set)
    for r in rows:
        for k, ids in r["ids"].items():
            byk[k].add(tuple(ids))
    print("- distinct id sequences per prompt across all runs and arms (1 = all equal): " + ", ".join(f"{k}: {len(v)}" for k, v in sorted(byk.items())))
    print("- reply lengths (prompt kind, tokens: replies): " + str(dict(cnt)))
    solos = [json.load(open(f)) for f in sorted(glob.glob(raw + "/q4-solo-*.json"))]
    if solos:
        print("- solo references across the servers: " + ", ".join(f"{k}: {len({tuple(s[k]) for s in solos})} distinct" for k in solos[0]))


if __name__ == "__main__":
    herd(); q4()
