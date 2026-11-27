"""Real-prompt serve sweep summary (lane ink-hip-bpf, window 10): per rep and level, serve_bitwise.py's meta (N real prompts at once,
128 forced tokens: aggregate = completion tokens / wall from the first send to the last reply, prefill included; request latency
median / max; prompt tokens) next to the server-histogram ITL / step buckets of that span (itl_seg.py, levels file), then the mean
per level; rep 1's replies (no nonce) compared with the solo server's (window 9).  python3 real_sum.py OUT_DIR VISIT SOLO_JSON"""
import collections
import glob
import json
import os
import re
import statistics
import subprocess
import sys

out, visit, solo = sys.argv[1:4]
here = os.path.dirname(os.path.abspath(__file__))
seg = {}
p = subprocess.run([sys.executable, os.path.join(here, "itl_seg.py"), out, visit], capture_output=True, text=True)
for line in p.stdout.splitlines():
    d = json.loads(line)
    if d.get("real"):
        seg[(d["rep"], d["streams"])] = d
agg = collections.defaultdict(list)
print("| rep | N at once | prompt tokens | aggregate tok/s | request latency med / max s | ITL p50 / p99 bucket ms | step p50 / p99 bucket ms | passes by rows | cached |")
print("|---|---|---|---|---|---|---|---|---|")
rows = []
for f in glob.glob(os.path.join(out, f"{visit}-real-r*-n*.meta.json")):
    m = re.search(r"-real-r(\d+)-n(\d+)\.meta\.json$", f)
    rows.append((int(m.group(1)), int(m.group(2)), json.load(open(f))))
for r, n, d in sorted(rows):
    s = seg.get((r, n), {})
    agg[n].append(d["aggregate_tok_s"])
    print(f"| {r} | {n} | {d['prompt_tokens']} | {d['aggregate_tok_s']} | {d['latency_s_median']} / {d['latency_s_max']} | "
          f"{s.get('itl_ms_p50')} / {s.get('itl_ms_p99')} | {s.get('step_ms_p50')} / {s.get('step_ms_p99')} | {s.get('decode_rows')} | "
          f"{d['cached_tokens']} |")
print()
for n, v in sorted(agg.items()):
    print(f"N {n}: aggregate mean {statistics.mean(v):.1f} tok/s over {len(v)} rep(s) ({' / '.join(str(x) for x in v)})")
print()
for f in sorted(glob.glob(os.path.join(out, f"{visit}-real-r1-n*.json"))):
    if f.endswith(".meta.json"):
        continue
    c = subprocess.run([sys.executable, os.path.join(here, "serve_bitwise.py"), "compare", solo, f], capture_output=True, text=True)
    print(f"rep 1 {os.path.basename(f)} vs solo: {c.stdout.strip().splitlines()[-1] if c.stdout.strip() else c.stderr.strip()[-200:]}")
