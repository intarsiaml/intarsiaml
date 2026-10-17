#!/usr/bin/env python3
"""M3 stage 0: project batched speculative verify from d793300 iteration traces and the measured pass costs.

  m3v_project.py trace  --dir TRACEDIR --cost COST.json [--ctx 512] [--corr g0b|meas|none] [--rmax 6] [--out OUT.json]
  m3v_project.py sim    ... (open-ended, gate-off traces; see below)

`trace`: for every `on` run in TRACEDIR (server-on-R.log with INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1, m3-on-R.json from
tools/cbm3.py) rebuild the scheduler iterations: DecodePhase (b decoding), the `spec:` lines printed by prepare (one solo
verify pass each, q = 1 + drafted rows) and the Decode event (rows of the batched step = plain rows). Then
  today(it)   = sum V(q) + P(plain)                       (P(1) = the solo step)
  batched(it) = first-fit-decreasing packing into passes of <= rmax rows, each pass costed as in the method section.
The aggregate is judged on the wall: projected wall = measured wall - (today - batched), tokens unchanged.
"""
import argparse, collections, json, re, statistics, sys

RMAX = 6
DEC = re.compile(r"DecodePhase \{ iter: (\d+), b: (\d+), spec_allowed: (\w+) \}")
EVD = re.compile(r"Decode \{ iter: (\d+), ids: \[([^\]]*)\], spec_allowed: (\w+) \}")
SPEC = re.compile(r"^spec: token (\d+) pos (\d+) drafted (\d+) accepted (\d+) emits (\d+)")


def parse_iterations(path):
    """-> list of dict(iter, b, groups=[(d, a, e)], plain=int). Only iterations with decoding sequences."""
    its, cur = [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = DEC.search(line)
            if m:
                cur = {"iter": int(m.group(1)), "b": int(m.group(2)), "groups": [], "plain": 0}
                its.append(cur)
                continue
            m = SPEC.match(line)
            if m and cur is not None:
                cur["groups"].append((int(m.group(3)), int(m.group(4)), int(m.group(5))))
                continue
            m = EVD.search(line)
            if m and cur is not None and int(m.group(1)) == cur["iter"]:
                ids = [x for x in m.group(2).split(",") if x.strip()]
                cur["plain"] = len(ids)
    return its


class Costs:
    def __init__(self, cost_json, ctx, corr):
        d = json.load(open(cost_json))
        c = str(ctx)
        self.P = {int(k[5:]): r["median"] for k, r in d["plain"][c].items()}
        self.V = {int(k[6:]): r["median"] for k, r in d["verify"][c].items()}
        self.V[1] = self.P[1]  # a 1-row "verify" is the solo step
        a = d["attn"][c]
        self.t1 = a["serving1"]["us"]
        self.tq = {int(k[4:]): r["us"] for k, r in a.items() if k.startswith("rows")}
        self.corr = corr

    def attn_corr(self, q):
        """ms saved when a q-row group replaces q single-row attention launches in a plain R-row step."""
        if q < 2 or self.corr == "none":
            return 0.0
        if self.corr == "g0b":
            return 36 * (q * self.t1 - self.tq[q]) / 1000.0
        if self.corr == "meas":
            return max(0.0, self.P[q] - self.V[q])
        raise ValueError(self.corr)

    def pass_cost(self, groups, plain):
        """One pass holding `groups` (list of q) and `plain` single rows."""
        r = sum(groups) + plain
        if r == 0:
            return 0.0
        if not groups:
            return self.P[plain]
        if len(groups) == 1 and plain == 0:
            return self.V[groups[0]]
        return self.P[r] - sum(self.attn_corr(q) for q in groups)


def today_cost(cs, groups, plain):
    return sum(cs.V[1 + d] for d, a, e in groups) + (cs.P[plain] if plain else 0.0)


def pack(qs, plain, rmax):
    """First-fit-decreasing: groups into bins of rmax rows, then plain rows into the free rows (rest into new bins)."""
    bins = []  # [groups, rows]
    for q in sorted(qs, reverse=True):
        for b in bins:
            if b[1] + q <= rmax:
                b[0].append(q)
                b[1] += q
                break
        else:
            bins.append([[q], q])
    left = plain
    for b in bins:
        take = min(left, rmax - b[1])
        b.append(take)
        left -= take
    out = [(b[0], b[2]) for b in bins]
    while left > 0:
        take = min(left, rmax)
        out.append(([], take))
        left -= take
    return out


def batched_cost(cs, groups, plain, rmax=RMAX):
    qs = [1 + d for d, a, e in groups]
    return sum(cs.pass_cost(g, p) for g, p in pack(qs, plain, rmax))


def kcap_variant(cs, groups, plain, rmax=RMAX):
    """Labelled approximation, not the k = 4 design: shrink drafts (largest group first) until all rows fit one pass; accepted
    becomes min(a, d'), the emitted tokens min(a, d') + 1. Returns (cost of the single pass, tokens emitted)."""
    ds = [d for d, a, e in groups]
    while sum(1 + d for d in ds) + plain > rmax and any(d > 0 for d in ds):
        i = max(range(len(ds)), key=lambda j: ds[j])
        ds[i] -= 1
    toks = sum(min(a, d2) + 1 for (d, a, e), d2 in zip(groups, ds)) + plain
    qs = [1 + d for d in ds if d > 0]
    plain2 = plain + sum(1 for d in ds if d == 0)
    return sum(cs.pass_cost(g, p) for g, p in pack(qs, plain2, rmax)), toks


def split_by_scenario(its_t, j):
    """Assign iterations (with their DecodePhase time in ms) to the scenarios of one tools/cbm3.py json by request windows."""
    win = {}
    for name, sc in j["scenarios"].items():
        reqs = sc["reqs"]
        lo = min(r["t_send"] for r in reqs)
        hi = max(r["t_end"] for r in reqs if r.get("t_end"))
        win[name] = (lo * 1000 - 50, hi * 1000 + 50)
    out = collections.defaultdict(list)
    for t, it in its_t:
        for name, (lo, hi) in win.items():
            if lo <= t <= hi:
                out[name].append(it)
                break
    return out


def parse_iterations_t(path):
    TS = re.compile(r"cb-trace t=(\d+) DecodePhase")
    its, cur = [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = DEC.search(line)
            if m:
                t = int(TS.search(line).group(1))
                cur = {"iter": int(m.group(1)), "b": int(m.group(2)), "groups": [], "plain": 0}
                its.append((t, cur))
                continue
            m = SPEC.match(line)
            if m and cur is not None:
                cur["groups"].append((int(m.group(3)), int(m.group(4)), int(m.group(5))))
                continue
            m = EVD.search(line)
            if m and cur is not None and int(m.group(1)) == cur["iter"]:
                cur["plain"] = len([x for x in m.group(2).split(",") if x.strip()])
    return its


def project_scenario(cs_by_corr, its, rmax):
    """Modelled decode GPU time (ms) of one scenario's iterations: today, batched per correction, kcap variant."""
    res = {"iters": len(its), "iters_with_verify": 0, "passes_today": 0, "pass_hist": collections.Counter()}
    cs0 = next(iter(cs_by_corr.values()))
    res["today_ms"] = 0.0
    res["batched_ms"] = {k: 0.0 for k in cs_by_corr}
    res["kcap_ms"] = 0.0
    res["kcap_tokens"] = 0
    res["kcap_orig_tokens"] = 0
    res["accepted"] = []
    res["comp"] = collections.Counter()
    for it in its:
        g, p = it["groups"], it["plain"]
        if not g and not p:
            continue
        if g:
            res["iters_with_verify"] += 1
            res["accepted"] += [(d, a) for d, a, e in g]
        qs = tuple(sorted((1 + d for d, a, e in g), reverse=True))
        res["comp"][(qs, p)] += 1
        res["passes_today"] += len(g) + (1 if p else 0)
        res["today_ms"] += today_cost(cs0, g, p)
        npass = len(pack([1 + d for d, a, e in g], p, rmax))
        res["pass_hist"][npass] += 1
        for k, cs in cs_by_corr.items():
            res["batched_ms"][k] += batched_cost(cs, g, p, rmax)
        c, toks = kcap_variant(cs0, g, p, rmax)
        res["kcap_ms"] += c
        res["kcap_tokens"] += toks
        res["kcap_orig_tokens"] += sum(a + 1 for d, a, e in g) + p
    res["comp"] = {f"{list(k[0])}+{k[1]}": v for k, v in sorted(res["comp"].items(), key=lambda kv: -kv[1])}
    res["pass_hist"] = dict(res["pass_hist"])
    return res


def hypo_aligned(cs, res, nstreams, kcap):
    """HYPOTHETICAL, outside the specified design (labelled as such in the doc): every pass is a pass of `nstreams` verify groups
    of 1 + kcap rows (R = nstreams x (1 + kcap) <= 6), all streams aligned every pass. A pass that accepted `a` drafts at k = 4
    is split into ceil((a + 1) / (kcap + 1)) capped passes of the same tokens (a matched run continues across the cut).
    Returns decode rate ratio against today's modelled rate for the same tokens."""
    acc = res["accepted"]
    toks = sum(a + 1 for d, a in acc)
    passes = sum(-(-(a + 1) // (kcap + 1)) for d, a in acc)
    t_per_pass = toks / passes
    q = 1 + kcap
    cost = cs.pass_cost([q] * nstreams, 0)
    rate_hyp = nstreams * t_per_pass / cost  # tokens per ms
    rate_today = (res["kcap_orig_tokens"]) / res["today_ms"]
    return {"tokens_per_capped_pass": t_per_pass, "pass_ms": cost, "rate_hyp": rate_hyp, "rate_today": rate_today,
            "ratio": rate_hyp / rate_today}


def run_trace(args):
    import glob, os
    corrs = {c: Costs(args.cost, args.ctx, c) for c in ("g0b", "meas", "none")}
    out = {}
    for jpath in sorted(glob.glob(os.path.join(args.dir, f"m3-{args.arm}-*.json"))):
        rnd = re.search(rf"m3-{args.arm}-(\d+)\.json", jpath).group(1)
        lpath = os.path.join(args.dir, f"server-{args.arm}-{rnd}.log")
        if not os.path.exists(lpath):
            continue
        j = json.load(open(jpath))
        its = split_by_scenario(parse_iterations_t(lpath), j)
        offp = os.path.join(args.dir, f"m3-off-{rnd}.json")
        joff = json.load(open(offp)) if os.path.exists(offp) else None
        for name, its_s in its.items():
            sc = j["scenarios"][name]
            if "wall_s" not in sc:
                continue
            r = project_scenario(corrs, its_s, args.rmax)
            r["wall_s"], r["tokens"], r["agg"] = sc["wall_s"], sc["tokens"], sc["agg_tok_s"]
            r["spec_counters"] = sc.get("counters_delta")
            r["measured_off_agg"] = joff["scenarios"][name]["agg_tok_s"] if joff and name in joff["scenarios"] else None
            for k in corrs:
                saved = (r["today_ms"] - r["batched_ms"][k]) / 1000
                r[f"proj_wall_{k}"] = sc["wall_s"] - saved
                r[f"proj_agg_{k}"] = sc["tokens"] / (sc["wall_s"] - saved)
            # kcap variant: iteration rate approximation, wall scaled by the modelled change in (cost per token)
            if r["kcap_tokens"]:
                today_per_tok = r["today_ms"] / max(1, r["kcap_orig_tokens"])
                kcap_per_tok = r["kcap_ms"] / r["kcap_tokens"]
                saved = (today_per_tok - kcap_per_tok) * r["kcap_orig_tokens"] / 1000
                r["proj_wall_kcap"] = sc["wall_s"] - saved
                r["proj_agg_kcap"] = sc["tokens"] / (sc["wall_s"] - saved)
            if name.startswith("grounded-n"):
                n = int(name[-1])
                kc = 2 if n == 2 else 1
                h = hypo_aligned(corrs["g0b"], r, n, kc)
                r["hypo"] = h
                saved = r["today_ms"] / 1000 * (1 - 1 / h["ratio"])
                r["proj_agg_hypo"] = sc["tokens"] / (sc["wall_s"] - saved)
            r.pop("accepted", None)
            out.setdefault(name, {})[rnd] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["trace"])
    ap.add_argument("--dir")
    ap.add_argument("--cost")
    ap.add_argument("--ctx", default="512")
    ap.add_argument("--corr", default="g0b")
    ap.add_argument("--rmax", type=int, default=6)
    ap.add_argument("--out")
    ap.add_argument("--arm", default="on")
    args = ap.parse_args()
    res = run_trace(args)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1, default=str)
    for name, rr in res.items():
        print(f"== {name}")
        for rnd, r in sorted(rr.items()):
            off = r["measured_off_agg"]
            line = (f"r{rnd}: wall {r['wall_s']:.2f}s agg {r['agg']:.1f} (off {off and round(off,1)}) iters {r['iters']} with-verify "
                    f"{r['iters_with_verify']}; today-model {r['today_ms']/1000:.2f}s; batched-model g0b {r['batched_ms']['g0b']/1000:.2f}s "
                    f"meas {r['batched_ms']['meas']/1000:.2f}s none {r['batched_ms']['none']/1000:.2f}s -> agg g0b {r['proj_agg_g0b']:.1f} "
                    f"meas {r['proj_agg_meas']:.1f} none {r['proj_agg_none']:.1f}")
            if "proj_agg_hypo" in r:
                h = r["hypo"]
                line += f" HYPO-aligned-kcap {r['proj_agg_hypo']:.1f} (x{h['ratio']:.2f} decode, {h['tokens_per_capped_pass']:.2f} tok/pass, pass {h['pass_ms']:.1f} ms)"
            if "proj_agg_kcap" in r:
                line += f" kcap {r['proj_agg_kcap']:.1f}"
            print(line)



if __name__ == "__main__":
    main()
