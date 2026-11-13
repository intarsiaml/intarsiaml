#!/usr/bin/env python3
"""Run localeval's ITL math (client.itl_stats) over sse_probe marks files: itl_localeval.py FILE.marks.json ...
Run with: cd /opt/code/deepeval && uv run python <this file> ..."""
import json, sys
sys.path.insert(0, "/opt/code/deepeval/src")
from localeval.client import itl_stats
for f in sys.argv[1:]:
    d = json.load(open(f)); m = [tuple(x) for x in d["marks"]]
    g = sorted((b[0] - a[0]) * 1e3 for a, b in zip(m, m[1:]))
    print(f"{f.rsplit('/',1)[-1]}: chunks={len(m)} tokens={d['n_tokens']} localeval_itl={itl_stats(m, d['n_tokens'])} "
          f"raw_chunk_gap_ms p50={g[len(g)//2]:.1f} p99={g[int(len(g)*.99)]:.1f}")
