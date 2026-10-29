#!/usr/bin/env python3
"""Lane msd: profile load only (NOT a benchmark; timing numbers come from localeval). N concurrent streaming chat requests with
the same ~1k-token prompt, M output tokens; prints per-stream TTFT / end (s) so the nsys timeline can be split into phases.
  burst.py <port> <streams> <gen> [rounds]"""
import json, os, sys, threading, time, urllib.request
port, n, gen = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 1
tag = sys.argv[5] if len(sys.argv) > 5 else ''  # lane srv16: the warm-up's tag, so it shares no prefix with the burst
corpus = open('/home/nvidia/intarsiaml-stream16/src/tools/prompts/corpus_docs.txt').read()
body = corpus[:3700]
DISTINCT = os.environ.get('DISTINCT') == '1'  # stream i gets its own 3700-char slice: different tokens, so different expert routing

def one(i, rnd, out):
    t0 = time.time()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps({
        'model': 'm', 'stream': True, 'max_tokens': gen, 'temperature': 0, 'ignore_eos': True, 'min_tokens': gen,
        'messages': [{'role': 'user', 'content': f'[{tag}{rnd}-{i}] ' + (corpus[(i * 3900) % max(1, len(corpus) - 3700):][:3700] if DISTINCT else body)}]}).encode(), {'content-type': 'application/json'})
    ttft = None; toks = 0
    for line in urllib.request.urlopen(req, timeout=600):
        if line.startswith(b'data: {'):
            d = json.loads(line[6:])
            if d.get('choices') and d['choices'][0].get('delta', {}).get('content'):
                toks += 1
                if ttft is None: ttft = time.time() - t0
    out[i] = (ttft, time.time() - t0, toks)

for rnd in range(rounds):
    out = [None] * n; ts = [threading.Thread(target=one, args=(i, rnd, out)) for i in range(n)]
    t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
    print(f'round {rnd} streams {n} wall {time.time() - t0:.2f}s ttft_med {sorted(o[0] for o in out)[n // 2]:.2f} ttft_max {max(o[0] for o in out):.2f} chunks {[o[2] for o in out]}', flush=True)
    time.sleep(2)
