#!/usr/bin/env python3
"""Lane cmtp: streaming greedy chat client for serve A/B and nsys windows (not a benchmark harness for quality).
  gen.py <port> <streams> <gen_tokens> <prompt_file> [chars] [reps] [tag]
N concurrent streams (distinct '[tag-rep-i]' prefix so prompts differ), forced <gen_tokens> (min_tokens), reports per stream
TTFT, decode tok/s (tokens after the first over time after the first) and the aggregate. One JSON line per rep on stdout."""
import json, sys, threading, time, urllib.request
port, n, gen, pf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
chars = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != '-' else 0
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 1
tag = sys.argv[7] if len(sys.argv) > 7 else 'g'
body = open(pf).read()
if chars: body = body[:chars]

def one(i, rep, out):
    t0 = time.time()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps({
        'model': 'm', 'stream': True, 'max_tokens': gen, 'min_tokens': gen, 'ignore_eos': True, 'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False},
        'messages': [{'role': 'user', 'content': f'[{tag}-{rep}-{i}] ' + body}]}).encode(), {'content-type': 'application/json'})
    ttft = None; t1 = None; toks = 0
    for line in urllib.request.urlopen(req, timeout=900):
        if line.startswith(b'data: {'):
            d = json.loads(line[6:])
            if d.get('choices') and d['choices'][0].get('delta', {}).get('content'):
                toks += 1
                if ttft is None: ttft = time.time() - t0; t1 = time.time()
    end = time.time()
    out[i] = (ttft, (toks - 1) / (end - t1) if t1 and end > t1 else 0.0, toks, end - t0)

for rep in range(reps):
    out = [None] * n; ts = [threading.Thread(target=one, args=(i, rep, out)) for i in range(n)]
    t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.time() - t0
    print(json.dumps({'rep': rep, 'streams': n, 'wall': round(wall, 3), 'ttft_med': round(sorted(o[0] for o in out)[n // 2], 3),
                      'dec_toks_per_s_per_stream_med': round(sorted(o[1] for o in out)[n // 2], 2),
                      'chunks': [o[2] for o in out]}), flush=True)
