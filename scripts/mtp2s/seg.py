#!/usr/bin/env python3
"""Lane mtp2s Step 1: split a node-level nsys sqlite export of a served decode window into passes and attribute GPU time.

  seg.py <file.sqlite> [--names]

A pass ends where the host downloads its tokens: every device-to-host copy closes the segment of kernels launched since the
previous one. Per segment: kernel count, captured (graph nodes) or eager, GPU busy (union of kernel intervals), span (first
kernel start to the copy's end) and the host gap before it (previous copy's end to its first kernel). Segments are grouped by
(captured, kernel count) and each group gets a per-op-class table (mean ms per segment). --names prints the top kernel names
per group (to check the class regexes)."""
import collections, re, sqlite3, statistics as st, sys

CLASSES = [  # first match wins; applied to the demangled/short kernel name
    ("moe_experts", r"moe_experts|moe_decode|moe_gate_up|moe_down|experts"),
    ("moe_router", r"moe_topk|topk_softmax|router"),
    ("gdn", r"gdn|delta|gated_delta|conv1d|short_conv|l2norm"),
    ("attention", r"attn|flash|sdpa|rope|kv_|paged|softmax_rows"),
    ("lm_head/argmax", r"argmax|lm_head|logits|top_k|topk_rows|sample"),
    ("proj_gemv", r"gemv|matmul|gemm|mma|fp8|nvfp4|mxfp4"),
    ("norm/elementwise", r"norm|silu|sigmoid|add|mul|copy|embed|gather|scatter|cast|fill|swiglu"),
]
RX = [(c, re.compile(r, re.I)) for c, r in CLASSES]


def cls(name):
    for c, r in RX:
        if r.search(name):
            return c
    return "other"


def main():
    path = sys.argv[1]
    show = "--names" in sys.argv
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    ks = db.execute(
        "select start, end, coalesce(demangledName, shortName), shortName, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ).fetchall()
    cps = db.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2 order by start").fetchall()
    ks = [(s, e, names.get(sn, str(sn)), g is not None) for s, e, _dn, sn, g in ks]
    # segments: kernels whose start is before a D2H copy's start, after the previous copy
    segs, ki, prev_end = [], 0, None
    for cs, ce, nb in cps:
        cur = []
        while ki < len(ks) and ks[ki][0] < cs:
            cur.append(ks[ki]); ki += 1
        if cur:
            busy, cend = 0, cur[0][0]
            for s, e, *_ in cur:
                s2 = max(s, cend)
                if e > s2:
                    busy += e - s2; cend = e
            graph = sum(k[3] for k in cur) > len(cur) / 2
            segs.append(dict(n=len(cur), graph=graph, busy=busy, span=ce - cur[0][0], gap=(cur[0][0] - prev_end) if prev_end else None,
                             start=cur[0][0], end=ce, ks=cur, dl=nb))
        prev_end = ce
    if not segs:
        print("no segments"); return
    t0, t1 = segs[0]["start"], segs[-1]["end"]
    print(f"{path}: {len(ks)} kernels, {len(cps)} D2H copies, {len(segs)} segments over {(t1 - t0) / 1e6:.1f} ms")
    # group by (graph, kernel count rounded)
    groups = collections.defaultdict(list)
    for s in segs:
        groups[(s["graph"], s["n"])].append(s)
    print("\n| captured | kernels | segments | span ms (mean) | GPU busy ms | host gap before ms (median) | share of window |")
    print("|---|---|---|---|---|---|---|")
    tot = t1 - t0
    for key in sorted(groups, key=lambda k: -sum(s["span"] for s in groups[k])):
        g = groups[key]
        if sum(s["span"] for s in g) < 0.005 * tot:
            continue
        gaps = [s["gap"] for s in g if s["gap"] is not None]
        print(f"| {'graph' if key[0] else 'eager'} | {key[1]} | {len(g)} | {st.mean(s['span'] for s in g) / 1e6:.3f} | "
              f"{st.mean(s['busy'] for s in g) / 1e6:.3f} | {(st.median(gaps) / 1e6 if gaps else 0):.3f} | {100 * sum(s['span'] + (s['gap'] or 0) for s in g) / tot:.1f} % |")
    # per-class table for the biggest groups
    big = sorted(groups, key=lambda k: -sum(s["span"] for s in groups[k]))[:6]
    allc = [c for c, _ in CLASSES] + ["other"]
    print("\nper op class, mean GPU kernel ms per segment (kernel durations, overlap not removed):\n")
    print("| captured / kernels | " + " | ".join(allc) + " | sum |")
    print("|---" * (len(allc) + 2) + "|")
    for key in big:
        g = groups[key]
        acc = collections.Counter()
        for s in g:
            for ks_, ke, nm, _ in s["ks"]:
                acc[cls(nm)] += ke - ks_
        n = len(g)
        print(f"| {'graph' if key[0] else 'eager'} / {key[1]} (x{n}) | " + " | ".join(f"{acc[c] / n / 1e6:.3f}" for c in allc) + f" | {sum(acc.values()) / n / 1e6:.3f} |")
    if show:
        for key in big:
            g = groups[key]
            acc = collections.Counter(); cnt = collections.Counter()
            for s in g:
                for ks_, ke, nm, _ in s["ks"]:
                    acc[nm] += ke - ks_; cnt[nm] += 1
            print(f"\n{'graph' if key[0] else 'eager'} / {key[1]} kernels (x{len(g)}): top names, ms per segment, launches per segment, class")
            for nm, t in acc.most_common(25):
                print(f"  {t / len(g) / 1e6:8.3f}  {cnt[nm] / len(g):6.1f}  {cls(nm):16s} {nm[:110]}")
    # the window's sequence of segment kinds (first 60), to see the iteration structure
    print("\nsequence (first 60 segments: G/E + kernels):")
    print(" ".join(f"{'G' if s['graph'] else 'E'}{s['n']}" for s in segs[:60]))


if __name__ == "__main__":
    main()
