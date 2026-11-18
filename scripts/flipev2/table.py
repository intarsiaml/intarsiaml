#!/usr/bin/env python3
"""flipev2: tables from raw/hs.jsonl, raw/age.jsonl, raw/dec.jsonl (whichever exist). usage: table.py <raw dir>"""
import json, sys, os, collections, glob, statistics as st

raw = sys.argv[1]
M = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
SD = lambda v: st.stdev(v) if len(v) > 1 else 0.0  # noqa: E731
NAME = {"N": "neither", "H": "herd", "S": "sjf", "B": "herd + sjf"}


def load(f):
    p = os.path.join(raw, f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else float("nan")


def rng(v, d=2):
    return f"{M(v):.{d}f} ({min(v):.{d}f} to {max(v):.{d}f})"


def hs():
    rows = load("hs.jsonl")
    if not rows:
        return
    conc = [r for r in rows if r["mode"] in ("herd8", "mix", "hs")]
    solo = [r for r in rows if r["mode"] == "solo"]
    ref = {}
    for r in solo:
        ref.setdefault((r["seed"], r["prompts"]), r["ids"])
    kind = lambda r: "hs" if r["mode"] == "hs" else "mix" if r["mode"] == "mix" else "herd"  # noqa: E731
    # ---- cell 1
    c1 = [r for r in conc if r["mode"] == "herd8"]
    print("## cell 1: 16 requests on a shared 16448-token prefix, `--max-batch 16` (mean over cells, range)\n")
    print("| herd | cells | TTFT sum s | TTFT p50 s | TTFT last s | wall s |\n|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in c1:
        g[r["arm"]].append(r)
    for a in "NHSB":
        if a in g:
            rs = g[a]
            print(f"| {'off' if a in 'NS' else 'on'} ({a}) | {len(rs)} | {rng([r['ttft_sum_s'] for r in rs])} | {rng([r['ttft_p50_s'] for r in rs])} | {rng([r['ttft_last_s'] for r in rs])} | {rng([r['wall_s'] for r in rs])} |")
    print("\nper seed, TTFT sum s (off / on): " + "; ".join(
        f"{s}: " + " / ".join(f"{M([r['ttft_sum_s'] for r in c1 if r['seed'] == s and r['arm'] == a]):.1f}" for a in "NH") for s in sorted({r['seed'] for r in c1})))
    # ---- cell 2
    c2 = [r for r in conc if r["mode"] == "hs"]
    print("\n## cell 2: 8 requests on a shared 16448-token prefix + one unrelated 1000-token prompt at +2 s, `--max-batch 16` (mean over cells, range)\n")
    print("| arm | cells | herd-8 TTFT sum s | herd-8 TTFT last s | short TTFT s | short end s (abs) | wall s | cached tokens of short |\n|---|---|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in c2:
        g[r["arm"]].append(r)
    for a in "NHSB":
        if a in g:
            rs = g[a]
            print(f"| {NAME[a]} | {len(rs)} | {rng([sum(r['ttft_s'][:8]) for r in rs])} | {rng([max(r['ttft_s'][:8]) for r in rs])} | {rng([r['ttft_s'][8] for r in rs])} | {rng([r['abs_end_s'][8] for r in rs])} | {rng([r['wall_s'] for r in rs])} | {sorted({r['cached_tokens'][8] for r in rs})} |")
    # ---- cell 4
    c4 = [r for r in conc if r["mode"] == "mix"]
    print("\n## cell 4: 4 requests on a 16448-token prefix + 2 unrelated 8192-token prompts, released together (thread order S U S S U S), `--max-batch 16`\n")
    print("per cell: TTFT of the two unrelated prompts (index 1 and index 4)\n")
    print("| arm | visit | seed | U1 (idx 1) s | U2 (idx 4) s | pair sum s | slower of two s | shared four (mean) s |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(c4, key=lambda r: (r["arm"], r["visit"], r["seed"])):
        t = r["ttft_s"]
        sh = [t[i] for i in (0, 2, 3, 5)]
        print(f"| {r['arm']} | {r['visit']} | {r['seed']} | {t[1]:.2f} | {t[4]:.2f} | {t[1] + t[4]:.2f} | {max(t[1], t[4]):.2f} | {M(sh):.2f} |")
    print("\nsummary (mean +- sd over cells)\n")
    print("| arm | cells | U1 s | U2 s | pair sum s | slower of two s | faster of two s | shared four mean s |\n|---|---|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in c4:
        g[r["arm"]].append(r)
    for a in "NHSB":
        if a in g:
            rs = g[a]; T = [r["ttft_s"] for r in rs]
            f = lambda v: f"{M(v):.2f} +- {SD(v):.2f}"  # noqa: E731
            print(f"| {NAME[a]} | {len(rs)} | {f([t[1] for t in T])} | {f([t[4] for t in T])} | {f([t[1] + t[4] for t in T])} | {f([max(t[1], t[4]) for t in T])} | {f([min(t[1], t[4]) for t in T])} | {f([M([t[i] for i in (0, 2, 3, 5)]) for t in T])} |")
    bv = collections.defaultdict(list)
    for r in c4:
        bv[(r["arm"], r["visit"])].append(r)
    print("\nper visit, pair sum s (mean over seeds): " + "; ".join(f"{a}/{v}: {M([r['ttft_s'][1] + r['ttft_s'][4] for r in rs]):.2f}" for (a, v), rs in sorted(bv.items())))
    # ---- ids
    print("\n## ids: every reply of every concurrent cell against the solo reply of the same prompt set (first visit that ran it)\n")
    tot = collections.Counter(); bad = collections.Counter(); errs = 0; ln = collections.Counter()
    for r in rows:
        errs += sum(1 for e in r["errors"] if e)
        if r["mode"] == "solo":
            key = (r["seed"], r["prompts"])
        else:
            key = (r["seed"], kind(r))
        for i, ids in enumerate(r["ids"]):
            ln[len(ids)] += 1
            if key not in ref:
                continue
            k = (r["mode"], kind(r) if r["mode"] != "solo" else "solo:" + r["prompts"], r["arm"])
            tot[k] += 1
            bad[k] += ids != ref[key][i]
    for k in sorted(tot):
        print(f"- {k[0]} / {k[1]} / arm {k[2]}: {tot[k] - bad[k]} / {tot[k]} equal")
    print(f"- errors: {errs}; reply lengths (tokens: replies): {dict(ln)}")
    print(f"- total: {sum(tot.values()) - sum(bad.values())} / {sum(tot.values())} equal")


def q4(name, label):
    rows = [r for r in load(name + ".jsonl")]
    if not rows:
        return
    err = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    void = [r for r in rows if not r.get("dec_alive_through_window", True)]
    rows = [r for r in rows if r.get("dec_alive_through_window", True)]
    key = lambda r: (r["d"], r["scn"], r["arm"])  # noqa: E731
    g = collections.defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    print(f"\n## {label}: ITL of the ongoing decodes while the 40K prompt prefills (event gaps in the window, pooled over runs)\n")
    print(f"runs dropped: {len(err)} with an error, {len(void)} void (a decoder finished before the long prompt's first token)\n")
    print("| decoders | scenario | arm | runs | gaps n | mean ms | p99 ms | max ms | gaps >200 ms | long TTFT s | short TTFT s (s1 / s2) | dec tokens in window | dec tok/s/stream | window s |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = {"F": 0, "S": 1}
    for k, rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], order.get(kv[0][2], int(kv[0][2]) if kv[0][2].isdigit() else 9))):
        d, scn, arm = k
        gw = [x for r in rs for x in r["gaps_window"]]
        sh = [r["short_ttft"] for r in rs if r["short_ttft"]]
        if sh and len(sh[0]) > 2:
            s = f"mean {M([M(list(x.values())) for x in sh]):.2f} / max {M([max(x.values()) for x in sh]):.2f} ({len(sh[0])} shorts)"
        else:
            s = f"{M([x['s1'] for x in sh]):.2f} / {M([x['s2'] for x in sh]):.2f}" if sh else "-"
        al = {"F": "fifo", "S": "sjf (age 8)"}.get(arm, f"sjf age {arm}")
        print(f"| {d} | {scn} | {al} | {len(rs)} | {len(gw)} | {M(gw) * 1e3:.1f} | {pct(gw, .99) * 1e3:.1f} | {max(gw) * 1e3:.1f} | {sum(x > .2 for x in gw)} | {rng([r['long_ttft'] for r in rs])} | {s} | {M([r['dec_tokens_in_window'] for r in rs]):.0f} | {M([r['dec_tok_s_in_window'] for r in rs]):.1f} | {M([r['window_s'] for r in rs]):.2f} |")
    print("\nper-run (arm visit rep): long TTFT / short TTFTs / dec tokens in window / max window gap ms\n")
    for k, rs in sorted(g.items()):
        for r in rs:
            st_ = r["short_ttft"]
            print(f"- d{k[0]} {k[1]} {r['arm']} {r.get('visit','')} rep{r['rep']}: {r['long_ttft']} / {' '.join(str(v) for v in st_.values()) or '-'} / {r['dec_tokens_in_window']} / {r['itl_window']['max'] * 1e3:.0f}")
    eq = collections.Counter(); tot = collections.Counter(); ln = collections.Counter()
    for r in rows:
        for k, v in (r["ids_eq_solo"] or {}).items():
            tot[r["arm"]] += 1; eq[r["arm"]] += bool(v)
        for k, n in r["ids_n"].items():
            ln[(k if k[0] != "d" else "d", n)] += 1
    print(f"\n{label} ids: " + "; ".join(f"arm {a}: {eq[a]} / {tot[a]} contended replies equal their own server's solo reply" for a in sorted(tot)))
    byk = collections.defaultdict(set)
    for r in rows:
        for k, ids in r["ids"].items():
            byk[k].add(tuple(ids))
    print("distinct id sequences per prompt over all runs and arms (1 = all equal): " + ", ".join(f"{k}: {len(v)}" for k, v in sorted(byk.items())))
    print("reply lengths: " + str(dict(ln)))


def mixd():
    rows = load("mixd.jsonl")
    if not rows:
        return
    conc = [r for r in rows if r["mode"] == "mix"]
    solo = [r for r in rows if r["mode"] == "solo"]
    ref = {}
    for r in solo:
        ref.setdefault(r["seed"], r["ids"])
    print("\n## cell 4, fixed arrival order (20 ms apart): 4 requests on a 16448-token prefix (S) + 2 unrelated 8192-token prompts (U), `--max-batch 16`\n")
    for arr, name in (("", "order A: S0 U1 S2 S3 U4 S5 (flipev's order)"), ("1,4,0,2,3,5", "order B: U1 U4 S0 S2 S3 S5 (the unrelated pair first)")):
        cs = [r for r in conc if r["arrival"] == arr]
        if not cs:
            continue
        print(f"### {name}\n")
        print("| request | kind | herd off TTFT s | herd on TTFT s | on - off s | off sd | on sd |\n|---|---|---|---|---|---|---|")
        kinds = ["S", "U", "S", "S", "U", "S"]
        arrpos = [int(x) for x in arr.split(",")] if arr else list(range(6))
        for i in range(6):
            a = [r["ttft_s"][i] for r in cs if r["arm"] == "N"]; b = [r["ttft_s"][i] for r in cs if r["arm"] == "H"]
            print(f"| idx {i} (arrives {arrpos.index(i) + 1}th) | {kinds[i]} | {M(a):.2f} | {M(b):.2f} | {M(b) - M(a):+.2f} | {SD(a):.2f} | {SD(b):.2f} |")
        for lab, f in (("pair sum of the two unrelated", lambda t: t[1] + t[4]), ("slower of the two unrelated", lambda t: max(t[1], t[4])), ("faster of the two unrelated", lambda t: min(t[1], t[4]))):
            a = [f(r["ttft_s"]) for r in cs if r["arm"] == "N"]; b = [f(r["ttft_s"]) for r in cs if r["arm"] == "H"]
            print(f"\n- {lab}: off {M(a):.2f} +- {SD(a):.2f} (n {len(a)}), on {M(b):.2f} +- {SD(b):.2f} (n {len(b)}), on - off {M(b) - M(a):+.2f}")
        print("\n  per visit, U pair sum / slower / faster (s): " + "; ".join(
            f"{a}/{v}: {M([r['ttft_s'][1] + r['ttft_s'][4] for r in cs if (r['arm'], r['visit']) == (a, v)]):.2f} / {M([max(r['ttft_s'][1], r['ttft_s'][4]) for r in cs if (r['arm'], r['visit']) == (a, v)]):.2f} / {M([min(r['ttft_s'][1], r['ttft_s'][4]) for r in cs if (r['arm'], r['visit']) == (a, v)]):.2f}"
            for a, v in sorted({(r['arm'], r['visit']) for r in cs})))
        print("\n  per cell (seed, arm, visit): U1 / U4 TTFT: " + "; ".join(f"{r['seed']} {r['arm']}/{r['visit']}: {r['ttft_s'][1]:.2f} / {r['ttft_s'][4]:.2f}" for r in sorted(cs, key=lambda r: (r['seed'], r['arm'], r['visit']))))
        print()
    tot = bad = 0
    for r in rows:
        for i, ids in enumerate(r["ids"]):
            tot += 1; bad += ids != ref[r["seed"]][i]
    print(f"ids: {tot - bad} / {tot} replies (concurrent and solo) equal the first solo reply of the same prompts; errors {sum(1 for r in rows for e in r['errors'] if e)}")


def diag():
    rows = load("diag.jsonl")
    if not rows:
        return
    print("\n## diag: 3 cold 16448-token prompts released together (FIFO prefill), TTFT per prompt s; the step between sorted TTFTs is one prompt's prefill + first token\n")
    print("| config | seed | TTFTs sorted s | steps s |\n|---|---|---|---|")
    for r in rows:
        t = sorted(x for x in r["ttft_s"] if x is not None)
        print(f"| {r['arm']} | {r['seed']} | {', '.join(f'{x:.2f}' for x in t)} | {', '.join(f'{b - a:.2f}' for a, b in zip([0] + t, t))} |")


if __name__ == "__main__":
    hs(); mixd(); diag(); q4("age", "cell 3 (AGE sensitivity, 2 decoders + 40K + 2 shorts, `--max-batch 4`)"); q4("dec", "cell 5 (3 / 4 decoders, `--max-batch 8`)"); q4("ageburst", "cell 3b (AGE with a burst of 12 shorts, 2 decoders, `--max-batch 16`)")
