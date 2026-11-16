#!/usr/bin/env python3
"""Lane mtp2s (2026-11-16): mixed 2-stream load for nsys: stream 0 decodes <gen> tokens in one request, stream 1 sends short requests
(48 tokens) back to back until stream 0 is done. Every start of a short request makes iterations where one sequence verifies (3 rows)
and the other takes a plain step (V+D), or a verify alone; the rest are the 2-sequence 6-row passes.
  gen3.py <port> <streams (ignored: 2)> <gen> [first_prompt_index]"""
import json, os, sys, threading, time, urllib.request
sys.argv_ = sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen2.py")).read().split("port, n, gen")[0])
port, gen = sys.argv[1], int(sys.argv[3])
first = int(sys.argv[4]) if len(sys.argv) > 4 else 0


def req(i, n):
    r = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps({
        'model': 'm', 'stream': True, 'max_tokens': n, 'min_tokens': n, 'ignore_eos': True, 'temperature': 0, 'seed': 2000 + i,
        'chat_template_kwargs': {'enable_thinking': False}, 'messages': [{'role': 'user', 'content': P[(first + i) % len(P)]}]}).encode(),
        {'content-type': 'application/json'})
    k = 0
    for line in urllib.request.urlopen(r, timeout=900):
        if line.startswith(b'data: {'):
            k += 1
    return k


done = threading.Event()


def short():
    i = 1
    while not done.is_set():
        req(i, 48); i += 1


t = threading.Thread(target=short); t0 = time.time(); t.start()
n = req(0, gen); done.set(); t.join()
print(json.dumps({'streams': 2, 'wall': round(time.time() - t0, 3), 'long_chunks': n}), flush=True)
