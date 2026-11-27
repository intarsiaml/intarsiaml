"""Serve window summary (lane ink-hip-bpf): per visit and rep, localeval speed's concurrency rows (aggregate tok/s, per-stream decode
tok/s median / min / max, TTFT median / max) next to the server-histogram ITL / step percentiles of the same level (itl_seg.py), then
the mean +- spread of the aggregate over reps.  python3 serve_sum.py OUT_DIR VISIT"""
import collections
import glob
import json
import os
import re
import statistics
import subprocess
import sys

out, visit = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
seg = {}
try:
    p = subprocess.run([sys.executable, os.path.join(here, "itl_seg.py"), out, visit], capture_output=True, text=True)
    for line in p.stdout.splitlines():
        d = json.loads(line)
        seg[(d["rep"], d["streams"])] = d
        if d["rep"] == 0:
            print(f"bitwise phase: {d['span_s']} s, ITL p50 / p99 bucket {d['itl_ms_p50']} / {d['itl_ms_p99']} ms, step p50 / p99 "
                  f"{d['step_ms_p50']} / {d['step_ms_p99']} ms, decode passes by rows {d['decode_rows']}")
except Exception as e:  # noqa: BLE001
    print(f"(itl_seg failed: {e})")
agg = collections.defaultdict(list)
print("| rep | streams | aggregate tok/s | per-stream decode tok/s med (min-max) | TTFT s med (max) | ITL p50 / p99 bucket ms | step p50 / p99 bucket ms | passes by rows |")
print("|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(out, f"{visit}-r*.result.json"))):
    rep = int(re.search(r"-r(\d+)\.result\.json$", f).group(1))
    d = json.load(open(f))
    res = d.get("speed") or d.get("result") or d
    conc = res.get("concurrency") if isinstance(res, dict) else None
    if conc is None:
        # localeval nests the sweep; find the first "concurrency" list
        def find(x):
            if isinstance(x, dict):
                if isinstance(x.get("concurrency"), list):
                    return x["concurrency"]
                for v in x.values():
                    r = find(v)
                    if r is not None:
                        return r
            if isinstance(x, list):
                for v in x:
                    r = find(v)
                    if r is not None:
                        return r
            return None
        conc = find(d) or []
    for c in conc:
        n = c["streams"]
        ps, tt = c.get("per_stream_decode_tok_s") or {}, c.get("ttft_s") or {}
        s = seg.get((rep, n), {})
        agg[n].append(c["aggregate_decode_tok_s"])
        print(f"| {rep} | {n} | {c['aggregate_decode_tok_s']} | {ps.get('median')} ({ps.get('min')}-{ps.get('max')}) | "
              f"{tt.get('median')} ({tt.get('max')}) | {s.get('itl_ms_p50')} / {s.get('itl_ms_p99')} | {s.get('step_ms_p50')} / "
              f"{s.get('step_ms_p99')} | {s.get('decode_rows')} |")
print()
for n, v in sorted(agg.items()):
    v = [x for x in v if x is not None]
    if v:
        print(f"streams {n}: aggregate mean {statistics.mean(v):.1f} tok/s over {len(v)} reps (min {min(v)}, max {max(v)})")
