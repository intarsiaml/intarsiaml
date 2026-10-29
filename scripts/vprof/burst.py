#!/usr/bin/env python3
"""Lane vprof: profile load against vLLM (NOT a benchmark). N concurrent streaming chat requests, ~1k-token prompt, M output tokens.
Same shape as scripts/step16/burst.py (prompt = first 3700 chars of corpus_docs.txt behind a per-stream tag; DISTINCT=1 gives each
stream its own slice) but talks to vLLM: model name from $MODEL, thinking off, ignore_eos.
  burst.py <port> <streams> <gen> <tag> <corpus>"""
import json, os, sys, threading, time, urllib.request
port, n, gen, tag, corpus_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
model = os.environ.get('MODEL', 'guidance')
corpus = open(corpus_path).read()
body = corpus[:3700]
DISTINCT = os.environ.get('DISTINCT') == '1'

def one(i, out):
    t0 = time.time()
    msg = f'[{tag}-{i}] ' + (corpus[(i * 3900) % max(1, len(corpus) - 3700):][:3700] if DISTINCT else body)
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps({
        'model': model, 'stream': True, 'max_tokens': gen, 'min_tokens': gen, 'temperature': 0, 'ignore_eos': True,
        'chat_template_kwargs': {'enable_thinking': False},
        'messages': [{'role': 'user', 'content': msg}]}).encode(), {'content-type': 'application/json'})
    ttft = None; toks = 0
    for line in urllib.request.urlopen(req, timeout=900):
        if line.startswith(b'data: {'):
            d = json.loads(line[6:])
            if d.get('choices'):
                dl = d['choices'][0].get('delta', {})
                if dl.get('content') or dl.get('reasoning_content') or dl.get('reasoning'):
                    toks += 1
                    if ttft is None: ttft = time.time() - t0
    out[i] = (ttft, time.time() - t0, toks)

out = [None] * n; ts = [threading.Thread(target=one, args=(i, out)) for i in range(n)]
t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
wall = time.time() - t0
print(f'burst {tag} streams {n} distinct {int(DISTINCT)} wall {wall:.2f}s ttft_med {sorted(o[0] for o in out)[n // 2]:.2f} ttft_max {max(o[0] for o in out):.2f} chunks {[o[2] for o in out]}', flush=True)
