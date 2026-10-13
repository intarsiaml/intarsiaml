#!/usr/bin/env python3
"""Apply the decision rule of scripts/phase1b-acceptance.sh to <dir>/<arm>-r<N>/*.json. Exit 0 PASS, 1 FAIL, 3 undecided."""
import glob, json, os, re, statistics, sys

def load(d):
    f = glob.glob(os.path.join(d, "*.json"))
    return json.load(open(f[0])) if f else None

def inflight(d):
    try:
        r, w = open(os.path.join(d, "inflight")).read().split()[:2]
        return int(r), int(w)
    except Exception:
        return None

def valid_round(root, arm, n):
    d = os.path.join(root, f"{arm}-r{n}")
    j = load(d)
    if j is None:
        return None, "no result"
    why = []
    inf = inflight(d)
    if inf is None: why.append("no in-flight samples")
    elif inf[0] > 1 or inf[1] > 0: why.append(f"in-flight seen running={inf[0]} waiting={inf[1]}")
    if arm == "ours-new":
        for row in j["rows"]:
            if row.get("skipped"): continue
            g = row.get("graph") or {}
            if not g.get("available"): why.append(f"size {row['target_prompt_tokens']}: no graph metrics")
            elif (g.get("replays") or 0) <= 0: why.append(f"size {row['target_prompt_tokens']}: 0 graph replays")
            elif (g.get("eager_steps") or 0) != 0 or (g.get("profiled_steps") or 0) != 0:
                why.append(f"size {row['target_prompt_tokens']}: eager={g.get('eager_steps')} profiled={g.get('profiled_steps')}")
    return (j, why)

def main():
    args = sys.argv[1:]
    count_only = args and args[0] == "--count-valid"
    root = args[-1]
    rounds = sorted({int(m.group(1)) for p in glob.glob(os.path.join(root, "*-r*")) if (m := re.search(r"-r(\d+)$", p))})
    valid, notes = [], []
    for n in rounds:
        ok = True
        res = {}
        for arm in ("ours-new", "vllm"):
            j, why = valid_round(root, arm, n)
            if j is None or why:
                ok = False; notes.append(f"round {n} {arm}: {why if j is not None else 'missing'}")
            res[arm] = j
        if ok: valid.append((n, res))
    if count_only:
        print(len(valid)); return 0
    sizes = [r["target_prompt_tokens"] for r in valid[0][1]["vllm"]["rows"]] if valid else []
    print(f"valid rounds: {[n for n, _ in valid]} of {rounds}")
    for x in notes: print("  invalid:", x)
    if not valid and any("graph" in x for x in notes):
        print("A/B FAILED: ours-new did not replay CUDA graphs (0 replays, no graph metrics, or eager/profiled decode steps): the numbers would be Phase 1a numbers"); return 1
    verdicts = {}
    for s in sizes:
        def med(arm, res):
            row = next(r for r in res[arm]["rows"] if r["target_prompt_tokens"] == s)
            return row["decode_tok_s"], row.get("decode_tok_s_reps")
        d, line = [], []
        for n, res in valid:
            (a, ar), (b, br) = med("ours-new", res), med("vllm", res)
            if a is None or b is None: continue
            d.append(a - b)
            line.append(f"r{n}: ours {a:.2f} vllm {b:.2f} d {a-b:+.2f}")
        old = []
        for n, _ in valid:
            j, _w = valid_round(root, "ours-old", n)
            if j:
                row = next((r for r in j["rows"] if r["target_prompt_tokens"] == s), None)
                if row and row["decode_tok_s"]: old.append(row["decode_tok_s"])
        if len(d) < 3: v = "INCONCLUSIVE"
        elif all(x >= 0 for x in d): v = "PASS"
        elif statistics.median(d) >= 0: v = "MARGINAL"
        else: v = "FAIL"
        verdicts[s] = v
        rng = f"d min/median/max {min(d):+.2f}/{statistics.median(d):+.2f}/{max(d):+.2f}" if d else "no data"
        print(f"size {s:>5}: {v:<12} {rng}; ours-old rounds {['%.1f' % x for x in old]}")
        for l in line: print("    " + l)
    vs = list(verdicts.values())
    if vs and all(v == "PASS" for v in vs): print("PHASE 1 SPEED GATE: PASS"); return 0
    if any(v == "FAIL" for v in vs): print("PHASE 1 SPEED GATE: FAIL"); return 1
    print("PHASE 1 SPEED GATE: NOT DECIDED (marginal or inconclusive sizes; report the numbers above)"); return 3

sys.exit(main())
