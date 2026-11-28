#!/usr/bin/env python3
"""Lane ink-metal-dtopk slot P: where a speculative pass's GPU time goes on Metal. Reads an INTARSIA_METAL_PROFILE=cb trace taken
with INTARSIA_METAL_COMMIT_EVERY=1 (one record per dispatch, the backend's labels) and the run's stderr (INTARSIA_SPEC_TRACE=1:
one `spec: dflash2 draft` line per drafting pass). Decode-phase records only (t <= 16: prefill chunks are larger). Per op (matmul-like
ops split by weight shape n x k) the GPU ms per pass, the share, and the row counts t seen.
  prof.py trace.jsonl run.err [top]"""
import collections, json, sys
trace, err = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 30
passes = sum(1 for l in open(err, errors="replace") if "spec: dflash2 draft flush" in l)
tot = collections.defaultdict(float); cnt = collections.Counter(); ts = collections.defaultdict(collections.Counter)
first = last = None; n_rec = 0
for l in open(trace):
    try:
        r = json.loads(l)
    except ValueError:
        continue
    if r.get("t", 0) > 16:
        continue
    key = r["op"] + (f" {r['n']}x{r['k']}" if r.get("bytes") else "")
    d = (r["e"] - r["s"]) * 1e3
    tot[key] += d; cnt[key] += 1; ts[key][r.get("t", 0)] += 1; n_rec += 1
    first = r["s"] if first is None else min(first, r["s"]); last = r["e"] if last is None else max(last, r["e"])
all_ms = sum(tot.values())
p = max(passes, 1)
print(f"passes {passes}, decode-phase dispatch records {n_rec} ({n_rec / p:.0f} a pass), GPU busy {all_ms / p:.2f} ms a pass, span {(last - first) * 1e3 / p:.2f} ms a pass")
print("| op (n x k) | ms / pass | share | dispatches / pass | rows t (count) |\n|---|---:|---:|---:|---|")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:top]:
    tt = ", ".join(f"{t}:{c}" for t, c in sorted(ts[k].items())[:6])
    print(f"| {k} | {v / p:.3f} | {100 * v / all_ms:.1f} % | {cnt[k] / p:.1f} | {tt} |")
