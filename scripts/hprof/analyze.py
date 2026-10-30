#!/usr/bin/env python3
"""Lane hprof: per-step, per-kernel analysis of a `rocprofv3 --kernel-trace` CSV (.csv or .csv.gz) of the batched 35B decode.

  analyze.py explore k_kernel_trace.csv.gz            what the trace holds: kernel names, grids, step candidates
  analyze.py steps   k_kernel_trace.csv.gz [--agent A] per-step table (kernels, wall, busy, rows) and the phases
  analyze.py report  k_kernel_trace.csv.gz [--agent A] [--rows B] [--json out.json]
                                                      per kernel group / kernel breakdown of the steady steps with B rows

A step ends at an `argmax` kernel (greedy sampling, temperature 0: one per decode step, none inside a step). Steps are the kernels of one
agent (GPU) between two argmax launches. The row count B of a step comes from the grid of a kernel whose grid scales with the rows
(ROWS_KERNEL, found by `explore`). Prefill chunks interleaved with decode show up as steps with a different kernel mix and are dropped.
"""
import collections, csv, gzip, json, re, statistics as st, sys

IDLE_NS = 1_000_000          # a hole this long between two launches of a step is idle time between requests, not part of the step


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    ag = collections.defaultdict(list)
    with op(path, "rt") as f:
        for r in csv.DictReader(f):
            ag[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"],
                                      int(r.get("Grid_Size_X", 0) or 0), int(r.get("Grid_Size_Y", 0) or 0), int(r.get("Grid_Size_Z", 0) or 0),
                                      int(r.get("Workgroup_Size_X", 0) or 0)))
    for k in ag:
        ag[k].sort()
    return ag


def steps_of(ks, anchor="argmax"):
    """Split one agent's sorted launches at the anchor kernel: list of lists (the anchor is the last kernel of its step)."""
    out, cur = [], []
    for k in ks:
        cur.append(k)
        if anchor in k[2]:
            # drop an idle lead-in: everything before the last > 1 ms hole between consecutive launches of this segment
            for j in range(len(cur) - 1, 0, -1):
                if cur[j][0] - cur[j - 1][1] > IDLE_NS:
                    cur = cur[j:]
                    break
            out.append(cur)
            cur = []
    return out


def short(n):
    n = re.sub(r"\(anonymous namespace\)::|HIP_vector_type<[^>]*>|ik_bf16|\s+", " ", n)
    return n.strip()[:110]


def busy_union(part):
    busy, cs, ce = 0, None, None
    for (b, e, *_r) in part:
        if ce is None or b > ce:
            if ce is not None:
                busy += ce - cs
            cs, ce = b, e
        else:
            ce = max(ce, e)
    return busy + (ce - cs if ce is not None else 0)


def explore(path):
    ag = load(path)
    for a, ks in sorted(ag.items()):
        print(f"\n== agent {a}: {len(ks)} launches, span {(ks[-1][1]-ks[0][0])/1e9:.2f} s")
        sts = steps_of(ks)
        c = collections.Counter(len(s) for s in sts)
        print("argmax-delimited segments:", len(sts), "by kernel count (count: segments):", dict(c.most_common(12)))
        by = collections.defaultdict(lambda: [0, 0, set()])
        for k in ks:
            x = by[k[2]]
            x[0] += k[1] - k[0]; x[1] += 1; x[2].add((k[3], k[4], k[5], k[6]))
        print("kernel | launches | total ms | distinct (grid x,y,z, wg x) up to 6")
        for n, (t, cnt, g) in sorted(by.items(), key=lambda x: -x[1][0])[:45]:
            print(f"  {short(n):<100} {cnt:7d} {t/1e6:9.1f}  {sorted(g)[:6]}{' ...' if len(g) > 6 else ''}")


def row_count(step):
    """Rows of a decode step: the Z grid of gdn_delta_rows_v2 (one block column per row) in a batched step, 1 for the solo step
    (gdn_conv_delta_kernel / gdn_delta_rows_v3 = one row), None when the segment has no GDN kernel (prefill chunk etc.)."""
    for k in step:
        if "gdn_delta_rows_v2" in k[2]:
            return k[5]
    for k in step:
        if "gdn_conv_delta_kernel" in k[2] or "gdn_delta_rows_v3" in k[2]:
            return 1
    return None


def summarize_steps(path, agent=None):
    ag = load(path)
    out = {}
    for a, ks in sorted(ag.items()):
        if agent is not None and a != agent:
            continue
        rows = []
        for i, st_ in enumerate(steps_of(ks)):
            if len(st_) < 100:
                continue
            wall = st_[-1][1] - st_[0][0]
            rows.append(dict(i=i, n=len(st_), B=row_count(st_), t0=st_[0][0], wall=wall, busy=busy_union(st_)))
        out[a] = rows
    return out


def phases(rows, gap_ns=1_500_000_000):
    """Split the step list into bursts separated by > gap_ns of no steps (the driver pauses between phases)."""
    ph, cur = [], []
    for r in rows:
        if cur and r["t0"] - (cur[-1]["t0"] + cur[-1]["wall"]) > gap_ns:
            ph.append(cur); cur = []
        cur.append(r)
    if cur:
        ph.append(cur)
    return ph


def cmd_steps(path, agent=None):
    for a, rows in summarize_steps(path, agent).items():
        print(f"== agent {a}: {len(rows)} decode-like steps (>= 100 kernels)")
        for pi, ph in enumerate(phases(rows)):
            c = collections.Counter((r["B"], r["n"]) for r in ph)
            print(f"-- burst {pi}: {len(ph)} steps, span {(ph[-1]['t0']+ph[-1]['wall']-ph[0]['t0'])/1e9:.2f} s")
            for (B, n), cnt in sorted(c.items(), key=lambda x: (-1 if x[0][0] is None else x[0][0], -x[1])):
                if cnt < 5:
                    continue
                w = [r["wall"] for r in ph if (r["B"], r["n"]) == (B, n)]
                bz = [r["busy"] for r in ph if (r["B"], r["n"]) == (B, n)]
                print(f"   B={B} kernels={n}: {cnt} steps, median wall {st.median(w)/1e6:.2f} ms, busy {st.median(bz)/1e6:.2f} ms")


def steady(path, agent, B):
    """Steps with B rows and the modal kernel count among them (a decode step with no prefill chunk mixed in)."""
    ag = load(path)
    ks = ag[agent]
    cand = [st_ for st_ in steps_of(ks) if len(st_) >= 100 and row_count(st_) == B]
    c = collections.Counter(len(x) for x in cand)
    n = c.most_common(1)[0][0]
    return [x for x in cand if len(x) == n]


def inst_table(sts):
    """Per (kernel, grid) instance over the steps: calls/step, median per-step total ns, mean ns per call. Instances are matched by their
    position in the step (name, grid, occurrence index) so the per-call times of one layer stay comparable."""
    acc = collections.defaultdict(list)
    for x in sts:
        seen = collections.Counter()
        for k in x:
            key = (k[2], k[3], k[4], k[5], k[6])
            acc[key].append(k[1] - k[0])
    nst = len(sts)
    rows = []
    for key, d in acc.items():
        rows.append(dict(name=key[0], grid=key[1:], calls=len(d) / nst, ns_step=sum(d) / nst, ns_call=sum(d) / len(d)))
    rows.sort(key=lambda r: -r["ns_step"])
    return rows


def cmd_dump(path, agent, B, top=80):
    sts = steady(path, agent, B)
    wall = st.median(x[-1][1] - x[0][0] for x in sts)
    busy = st.median(busy_union(x) for x in sts)
    print(f"B={B} agent={agent}: {len(sts)} steps of {len(sts[0])} kernels, median wall {wall/1e6:.3f} ms, busy {busy/1e6:.3f} ms")
    for r in inst_table(sts)[:top]:
        print(f"{r['ns_step']/1e3:9.1f} us/step {r['calls']:6.1f} calls {r['ns_call']/1e3:8.2f} us/call grid={r['grid']} {short(r['name'])}")


# ---- grouping and roofline ---------------------------------------------------------------------------------------------------------
BW_ROOF = 640.0          # GB/s, the roof the lane brief uses (l7 measured 585-627 GB/s on streaming reads)
NVFP4_B = 0.5625         # bytes per weight: 4-bit value + one e4m3 scale per 16 (the per-tensor f32 scale is negligible)


def group(n, gx, gy, gz):
    if "fp8_gemv" in n:
        return "dense fp8 projections"
    if "nvfp4_gemv_v2" in n:
        return "lm_head (NVFP4 GEMV)"
    if "argmax" in n:
        return "sampling (argmax)"
    if "moe_gate_up" in n or "moe_down_g2" in n:
        return "MoE routed experts"
    if "moe_block_kernel" in n:
        return "MoE fused block (B=1: router+top-k+routed+shared)"
    if "shared_glu" in n or "shared_down_combine" in n:
        return "shared expert"
    if "moe_topk" in n or "moe_group" in n or "moe_reduce" in n or "moe_combine" in n:
        return "MoE routing glue (top-k, group, reduce)"
    if "gemv_kernel<" in n or "gemv_pair_kernel" in n:
        return "small bf16 GEMVs (router, GDN b/a, shared-gate logit)"
    if re.search(r"ar2_kernel|ar2ll_kernel|allreduce|ar_norm", n):
        return "all-reduce (TP=2)"
    if "gdn_" in n or "gated_rmsnorm" in n:
        return "GatedDeltaNet (conv, delta rule, gated norm)"
    if any(t in n for t in ("splitkv", "combine_kernel", "rope_kernel", "kv_append", "gate_split", "sigmoid_mul", "gate_qk_norm", "flash")):
        return "attention (excl. projections)"
    if "rmsnorm" in n:
        return "norms / residual add"
    return "other (embedding, copies)"


def roof(n, gx, gy, gz, B, ctx, distinct):
    """(label, bytes read per call, flops per call) or None."""
    m = re.search(r"fp8_gemv_kernel<(\d+)", n)
    if m:
        M = int(m.group(1)); N = gx // 256 * 16; K = 4096 if N == 2048 else 2048
        return f"fp8 GEMV N={N} K={K} M={M}", N * K, 2 * M * N * K
    m = re.search(r"nvfp4_gemv_v2_kernel<(\d+)", n)
    if m:
        M = int(m.group(1)); N = gx // 32; K = 2048
        return f"lm_head NVFP4 N={N} M={M}", int(N * K * NVFP4_B), 2 * M * N * K
    if "moe_gate_up_g2" in n:
        return f"routed gate_up ({distinct} experts)", int(distinct * 1024 * 2048 * NVFP4_B), B * 8 * 2 * 1024 * 2048
    if "moe_down_g2" in n:
        return f"routed down ({distinct} experts)", int(distinct * 2048 * 512 * NVFP4_B), B * 8 * 2 * 2048 * 512
    if "shared_glu" in n:
        return "shared gate+up", int(1024 * 2048 * NVFP4_B), B * 2 * 1024 * 2048
    if "shared_down_combine" in n:
        return "shared down", int(2048 * 512 * NVFP4_B), B * 2 * 2048 * 512
    if "gdn_delta_rows_v2" in n:
        return "GDN delta rule (state r+w, fp32)", 2 * B * 32 * 128 * 128 * 4, B * 4 * 32 * 128 * 128
    if "splitkv" in n:
        return f"attn splitkv (ctx {ctx})", B * ctx * 2048, B * 16 * ctx * 256 * 4
    if "gemv_kernel<" in n:
        Nk = {512: 32 * 2, 4096: 256, 256: 1}.get(gx)
        if Nk:
            return {512: "GDN b/a bf16 GEMV", 4096: "router bf16 GEMV", 256: "shared-gate bf16 GEMV"}[gx], Nk * 2048 * 2, 2 * B * Nk * 2048
    return None


def step_stats(path, agent, B, follow=None):
    """Steady steps with B rows. follow = the agent whose steps (argmax-delimited) define the time windows; `agent` (e.g. the second
    rank at TP=2, which has no argmax) is then sliced by those windows. The gap list is only filled for the anchor agent."""
    ag = load(path)
    if follow:
        sts0, _ = step_stats(path, follow, B)
        ks = ag[agent]
        starts = [k[0] for k in ks]
        import bisect
        out = []
        for x in sts0:
            lo, hi = x[0][0], x[-1][1]
            i = bisect.bisect_left(starts, lo)
            j = bisect.bisect_right(starts, hi)
            out.append(ks[i:j])
        return out, []
    ks = ag[agent]
    allsteps = steps_of(ks)
    idx = {id(x): i for i, x in enumerate(allsteps)}
    cand = [x for x in allsteps if len(x) >= 100 and row_count(x) == B]
    n = collections.Counter(len(x) for x in cand).most_common(1)[0][0]
    sts = [x for x in cand if len(x) == n]
    # host gap between consecutive steps (next step's first launch - this step's last end), only for back-to-back steady steps
    gaps = []
    for x in sts:
        i = idx[id(x)]
        if i + 1 < len(allsteps) and len(allsteps[i + 1]) == n:
            gaps.append(allsteps[i + 1][0][0] - x[-1][1])
    return sts, gaps


def cmd_report(path, agent, Bs, distinct, ctx, out_md, out_json, follow=None):
    res = {}
    md = []
    for B in Bs:
        sts, gaps = step_stats(path, agent, B, follow)
        wall = [x[-1][1] - x[0][0] for x in sts]
        busy = [busy_union(x) for x in sts]
        g = collections.defaultdict(lambda: [0.0, 0.0])
        inst = collections.defaultdict(list)
        for x in sts:
            for k in x:
                gg = g[group(k[2], k[3], k[4], k[5])]
                gg[0] += (k[1] - k[0]) / len(sts)
                gg[1] += 1 / len(sts)
                inst[(k[2], k[3], k[4], k[5], k[6])].append(k[1] - k[0])
        w = st.median(wall)
        r = dict(B=B, steps=len(sts), kernels=len(sts[0]), wall_ms=w / 1e6, busy_ms=st.median(busy) / 1e6, idle_in_step_ms=(w - st.median(busy)) / 1e6,
                 inter_step_gap_us=(st.median(gaps) / 1e3 if gaps else None), groups={k: dict(ms=v[0] / 1e6, calls=v[1]) for k, v in g.items()}, inst=[])
        for key, d in sorted(inst.items(), key=lambda kv: -sum(kv[1])):
            n, gx, gy, gz, wg = key
            calls = len(d) / len(sts)
            us = sum(d) / len(d) / 1e3
            ro = roof(n, gx, gy, gz, B, ctx, distinct.get(B, 0))
            row = dict(name=short(n), grid=[gx, gy, gz, wg], group=group(n, gx, gy, gz), calls=calls, us_call=us, ms_step=us * calls / 1e3)
            if ro:
                lab, by, fl = ro
                row.update(label=lab, bytes=by, gbs=by / us / 1e3, pct_roof=100 * by / us / 1e3 / BW_ROOF, tflops=fl / us / 1e6)
            r["inst"].append(row)
        res[B] = r
        md.append(f"\n### B={B}: {r['steps']} steady steps, {r['kernels']} kernels/step, wall {r['wall_ms']:.2f} ms, kernel-busy {r['busy_ms']:.2f} ms, "
                  f"idle between launches {r['idle_in_step_ms']:.2f} ms, host gap to next step {r['inter_step_gap_us']:.0f} us\n" if gaps else f"\n### B={B}\n")
        md.append("| group | ms/step | share of wall | launches/step |\n|---|---:|---:|---:|")
        for k, v in sorted(r["groups"].items(), key=lambda kv: -kv[1]["ms"]):
            md.append(f"| {k} | {v['ms']:.2f} | {100*v['ms']/r['wall_ms']:.1f} % | {v['calls']:.0f} |")
        md.append(f"| idle between launches inside the step | {r['idle_in_step_ms']:.2f} | {100*r['idle_in_step_ms']/r['wall_ms']:.1f} % | |")
        md.append(f"| **step wall** | **{r['wall_ms']:.2f}** | 100 % | {r['kernels']} |")
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1)
    if out_md:
        open(out_md, "w").write("\n".join(md) + "\n")
    return res


def parse_kv(s):
    return {int(k): int(v) for k, v in (x.split("=") for x in s.split(","))} if s else {}


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "explore":
        explore(path)
    elif cmd == "steps":
        cmd_steps(path)
    elif cmd == "report":
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("path"); ap.add_argument("--agent", default="Agent 1")
        ap.add_argument("--rows", default="1,8,16"); ap.add_argument("--distinct", default="8=42,16=70"); ap.add_argument("--ctx", type=int, default=1200); ap.add_argument("--follow")
        ap.add_argument("--md"); ap.add_argument("--json")
        a = ap.parse_args()
        cmd_report(a.path, a.agent, [int(x) for x in a.rows.split(",")], parse_kv(a.distinct), a.ctx, a.md, a.json, a.follow)
    elif cmd == "dump":
        cmd_dump(path, sys.argv[3], int(sys.argv[4]))
    else:
        raise SystemExit("not implemented before the first trace: see explore")
