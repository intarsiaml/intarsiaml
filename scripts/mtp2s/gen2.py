#!/usr/bin/env python3
"""Lane mtp2s: N concurrent greedy chat streams (distinct natural prompts), exactly <gen> tokens each, against 127.0.0.1:<port>.
  gen2.py <port> <streams> <gen> [first_prompt_index]   prints one JSON line: per-stream decode tok/s and the aggregate."""
import json, os, sys, threading, time, urllib.request
P = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the sieve of Eratosthenes and why it is faster than trial division.",
    "Summarise the causes of the French Revolution in five paragraphs.",
    "Write a Rust function that merges two sorted vectors, with unit tests.",
    "Draft a polite email declining a meeting invitation and proposing two other times.",
]
port, n, gen = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
first = int(sys.argv[4]) if len(sys.argv) > 4 else 0


def one(i, out):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps({
        'model': 'm', 'stream': True, 'max_tokens': gen, 'min_tokens': gen, 'ignore_eos': True, 'temperature': float(os.environ.get('GEN_TEMP', '0')), 'seed': 2000 + i,
        'chat_template_kwargs': {'enable_thinking': False},
        'messages': [{'role': 'user', 'content': P[(first + i) % len(P)]}]}).encode(), {'content-type': 'application/json'})
    t1 = None; toks = 0
    for line in urllib.request.urlopen(req, timeout=900):
        if line.startswith(b'data: {'):
            d = json.loads(line[6:])
            if d.get('choices') and d['choices'][0].get('delta', {}).get('content'):
                toks += 1
                if t1 is None: t1 = time.time()
    end = time.time()
    out[i] = ((toks - 1) / (end - t1) if t1 and end > t1 else 0.0, toks, t1, end)


out = [None] * n
ts = [threading.Thread(target=one, args=(i, out)) for i in range(n)]
t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
lo, hi = max(o[2] for o in out), min(o[3] for o in out)
print(json.dumps({'streams': n, 'wall': round(time.time() - t0, 3), 'per_stream': [round(o[0], 2) for o in out],
                  'aggregate': round(sum(o[0] for o in out), 2), 'chunks': [o[1] for o in out], 'overlap_s': round(hi - lo, 3)}), flush=True)
