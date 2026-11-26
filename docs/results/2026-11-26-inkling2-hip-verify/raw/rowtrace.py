#!/usr/bin/env python3
"""Per-op-class GPU time of verify passes by row count, from a rocprofv3 kernel trace (csv) of the verify_rows_real probe in its
timing-only mode (INTARSIA_VERIFY_TIME_ONLY=1): the probe's cost loop runs REPS x [decode (1 row), verify 2, .., verify MOST], each
pass ends with exactly one argmax dispatch, and the loop is the end of the trace. So the last REPS * MOST argmax-delimited segments
are the loop's passes, in that order. The first SKIP reps are dropped (graph capture, warm-up).
usage: rowtrace.py trace.csv MOST REPS [SKIP=2] [--kernels]
Prints, per row count: dispatches, wall (first start to last end of the segment, i.e. GPU span), kernel time, and kernel time per
op class (median over the kept reps); with --kernels also every kernel's time per row count."""
import collections, csv, re, statistics, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path, most, reps = args[0], int(args[1]), int(args[2])
skip = int(args[3]) if len(args) > 3 else 2
show_k = "--kernels" in sys.argv
rows = sorted(csv.DictReader(open(path)), key=lambda r: int(r["Start_Timestamp"]))


def short(n):
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"\(.*$", "", n)
    n = re.sub(r"^void ", "", n)
    return n.replace(" [clone .kd]", "")[:72]


CLASSES = [
    ("experts nvfp4", r"ink_moe_(gu|dn)_kernel<0,"),
    ("experts mxfp8", r"ink_moe_(gu|dn)_kernel<1,"),
    ("route", r"moe_route|ink_route"),
    ("in-proj", r"in_proj"),
    ("unembed", r"vocab"),
    ("blend", r"blend|ink_ctl"),
    ("wo / dense gemv", r"gemv"),
    ("mix (conv / attn-res)", r"ink_row_kernel|conv_mix|attn_res|ink_mix"),
    ("attention", r"ink_attn|ink_prologue|attn_|splitkv|kv_append"),
    ("argmax", r"argmax"),
    ("norm / embed", r"rmsnorm|embedding|norm"),
    ("copy", r"copyBuffer|fillBuffer|rocclr"),
]


def cls(k, grid=0):
    # the router and the unembed can be the same GEMV kernel (decode: gemv_kernel<.., float>; verify: the Logits `_rows` twin):
    # the unembed's grid covers the vocab (> 1e6 threads), the router's 128 experts
    if re.search(r"gemv|logits", k) and not re.search(r"in_proj|ink_ctl", k):
        if grid >= 1_000_000:
            return "unembed"
        if "float" in k:
            return "route"
    for c, pat in CLASSES:
        if re.search(pat, k):
            return c
    return "other"


marks = [i for i, r in enumerate(rows) if "argmax" in r["Kernel_Name"]]
need = reps * most
if len(marks) < need + 1:
    sys.exit(f"only {len(marks)} argmax dispatches, need {need + 1}")
bounds = marks[-need - 1 :]
# per row count: list over reps of {class: us}, {kernel: us}, dispatch counts, spans
per = {q: [] for q in range(1, most + 1)}
for s in range(need):
    lo, hi = bounds[s] + 1, bounds[s + 1] + 1
    rep, q = divmod(s, most)
    q += 1
    if rep < skip:
        continue
    seg = rows[lo:hi]
    byc, byk = collections.defaultdict(float), collections.defaultdict(float)
    for r in seg:
        d = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
        g = int(r.get("Grid_Size_X") or 0)
        c = cls(short(r["Kernel_Name"]), g)
        k = short(r["Kernel_Name"]) + (f" [grid {g}]" if c in ("unembed", "route") and "gemv" in r["Kernel_Name"] else "")
        byk[k] += d
        byc[c] += d
    span = (int(seg[-1]["End_Timestamp"]) - int(seg[0]["Start_Timestamp"])) / 1e3
    per[q].append((len(seg), span, byc, byk))
# the segmentation check: every kept pass of a row count has the same dispatch count (else a boundary moved)
for q in per:
    ds = sorted({x[0] for x in per[q]})
    if len(ds) > 1:
        print(f"WARNING rows {q}: dispatch counts differ across reps: {ds}")

med = lambda v: statistics.median(v) if v else 0.0
classes = [c for c, _ in CLASSES] + ["other"]
print(f"trace {path}: {len(rows)} dispatches; loop {reps} reps x {most} passes, first {skip} reps dropped; ms per pass (median)")
hdr = f"{'rows':>4} {'disp':>5} {'span':>7} {'kernel':>7} " + " ".join(f"{c[:13]:>13}" for c in classes)
print(hdr)
for q in range(1, most + 1):
    v = per[q]
    if not v:
        continue
    disp = med([x[0] for x in v])
    span = med([x[1] for x in v]) / 1e3
    kt = med([sum(x[2].values()) for x in v]) / 1e3
    cs = [med([x[2].get(c, 0.0) for x in v]) / 1e3 for c in classes]
    print(f"{q:>4} {disp:>5.0f} {span:>7.2f} {kt:>7.2f} " + " ".join(f"{c:>13.3f}" for c in cs))
if show_k:
    ks = sorted({k for q in per for x in per[q] for k in x[3]}, key=lambda k: -med([x[3].get(k, 0) for x in per[most]]))
    print(f"\n{'kernel':72s} " + " ".join(f"{'r' + str(q):>8}" for q in range(1, most + 1)) + "   (us per pass, median)")
    for k in ks:
        print(f"{k:72s} " + " ".join(f"{med([x[3].get(k, 0) for x in per[q]]):>8.1f}" for q in range(1, most + 1)))
