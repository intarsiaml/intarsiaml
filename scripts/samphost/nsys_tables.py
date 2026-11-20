#!/usr/bin/env python3
"""samp64: kernel-level attribution of one nsys capture (sqlite export of a `nsys.sh` cell).
  python3 scripts/samphost/nsys_tables.py CAPTURE.sqlite [--json OUT.json]
The decode window starts after the last prefill attention kernel (name containing `prefill` or `flash`) and ends at the last kernel; kernels of the window are
grouped by category; GPU busy time is the union of the kernel intervals; D2H copies in the window are counted. `passes` is the number of captured-graph
launches or argmax/top-k kernel launches in the window, whichever the capture has (printed, to be checked by eye)."""
import json, re, sqlite3, sys, collections

CATS = [
    ("attention (KV read)", r"attn_decode|splitkv|attention|flash.*decode|attn_"),
    ("MoE experts + routing", r"moe_|topk_gate|expert"),
    ("matmul / GEMV (dense, lm head)", r"matmul|gemv|gemm|nvfp4|fp8|cutlass|cublas|wq_"),
    ("GatedDeltaNet / conv", r"gdn|conv"),
    ("sampling (top-k, argmax, softmax)", r"argmax|topk|top_k|sample|softmax|cumsum|sort"),
    ("norm / rope / elementwise / copy", r"rmsnorm|norm|rope|silu|sigmoid|add_|kv_append|gate_split|embedding|copy|memcpy|memset|elementwise"),
]

def cat_of(name):
    for c, rx in CATS:
        if re.search(rx, name, re.I):
            return c
    return "other"

def main():
    db = sqlite3.connect(sys.argv[1]); cur = db.cursor()
    names = dict(cur.execute("select id, value from StringIds"))
    ks = [(s, e, names.get(n, str(n)), g) for s, e, n, g in cur.execute("select start, end, demangledName, coalesce(graphNodeId, 0) from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    pf = [e for s, e, n, g in ks if re.search(r"prefill|flash", n, re.I)]
    t0 = max(pf) if pf else ks[0][0]
    win = [k for k in ks if k[0] >= t0]
    if not win:
        raise SystemExit("no kernels after the last prefill attention kernel")
    w0, w1 = win[0][0], win[-1][1]
    # union of kernel intervals
    busy, cur_e = 0, None
    for s, e, n, g in sorted(win):
        if cur_e is None or s > cur_e:
            busy += e - s; cur_e = e
        elif e > cur_e:
            busy += e - cur_e; cur_e = e
    by_cat = collections.Counter(); by_name = collections.Counter(); cnt = collections.Counter()
    for s, e, n, g in win:
        by_cat[cat_of(n)] += e - s; by_name[n] += e - s; cnt[n] += 1
    try:
        d2h = cur.execute("select count(*), coalesce(sum(bytes), 0), coalesce(sum("end" - start), 0) from CUPTI_ACTIVITY_KIND_MEMCPY where start >= ? and copyKind = 2", (w0,)).fetchone()
    except sqlite3.Error:
        d2h = (None, None, None)
    passes = {n: c for n, c in cnt.items() if re.search(r"argmax", n, re.I)}
    out = {"decode_wall_ms": (w1 - w0) / 1e6, "gpu_busy_ms": busy / 1e6, "kernels": len(win), "d2h_copies": d2h[0], "d2h_bytes": d2h[1], "d2h_ms": (d2h[2] or 0) / 1e6,
           "by_category_ms": {c: v / 1e6 for c, v in by_cat.most_common()}, "argmax_launches": passes,
           "top_kernels": [(n[:110], v / 1e6, cnt[n]) for n, v in by_name.most_common(25)]}
    print(json.dumps(out, indent=1))
    if "--json" in sys.argv:
        open(sys.argv[sys.argv.index("--json") + 1], "w").write(json.dumps(out, indent=1))

main()
