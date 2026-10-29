#!/usr/bin/env python3
"""Lane ARX2: prefill TTFT of localeval's filler prompt at N sentences (182 = the 4107-token prompt: chunks 2048 + 2048 + 11; 181 = 4084 tokens:
2048 + 2036) against a running server, REPS requests each, localeval's uuid nonce first (so the token counts are localeval's: 182 -> 4106-4108, 181 -> 4084; if 182 reads <= 4096 the run is invalid).
  cd /opt/code/deepeval && uv run python p4107.py <base url, http://host:port/v1> <model> [reps] [n ...]
"""
import statistics
import sys

from localeval.client import Client
from localeval.config import Target
from localeval.speed import INSTRUCTION, filler

url, model = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ns = [int(x) for x in sys.argv[4:]] or [182, 181]
c = Client(Target(name="p4107", url=url, model=model))
for n in ns:
    ttft, pre, pt = [], [], []
    for _ in range(reps):
        r = c.chat([{"role": "user", "content": filler(n, seed=4096) + INSTRUCTION}], max_tokens=1, nonce=True)
        if not r.ok:
            print(f"p4107 n={n}: ERROR {r.error}")
            continue
        ttft.append(r.ttft_s)
        pt.append(r.prompt_tokens)
        pre.append((r.prompt_tokens - (r.cached_tokens or 0)) / r.ttft_s)
    if ttft:
        print(f"p4107 n={n}: prompt={sorted(set(pt))} ttft={[round(t, 3) for t in ttft]} prefill tok/s={[round(p) for p in pre]} median={statistics.median(pre):.0f}")
