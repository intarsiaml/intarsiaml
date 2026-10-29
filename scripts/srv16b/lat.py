#!/usr/bin/env python3
"""Lane srv16b: latency of a served burst (NOT the throughput number; that comes from localeval). For each stream count, N concurrent
streaming chat requests at once, each its own ~1k-token prompt (localeval's size: ~994 tokens, so a row cap packs the same prompt
counts in both), GEN forced tokens. Client-side per-token arrival times give per-stream TTFT and every inter-token gap; the
server's /metrics deltas give its packed passes and its own ITL buckets.
  lat.py <port> <gen> <streams,...>     prints one line per burst"""
import json, re, sys, threading, time, urllib.request
port, gen, conc = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(',')]
corpus = open('/home/nvidia/intarsiaml-srv16b/src/tools/prompts/corpus_docs.txt').read()
CH = 3150  # ~990 prompt tokens with the chat template (checked from usage below)

def metrics():
    t = urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=30).read().decode()
    def g(name):
        m = re.search(rf'^{re.escape(name)} (\S+)$', t, re.M)
        return float(m.group(1)) if m else 0.0
    b = 'intarsia_batch_itl_seconds_bucket{le="%s"}'
    return {'packs': g('intarsia_batch_pack_passes_total'), 'chunks': g('intarsia_batch_packed_chunks_total'), 'n': g('intarsia_batch_itl_seconds_count'),
            'le025': g(b % '0.25'), 'le05': g(b % '0.5'), 'le1': g(b % '1')}

def one(i, tag, out):
    off = (i * 3300) % max(1, len(corpus) - CH)
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps({
        'model': 'm', 'stream': True, 'max_tokens': gen, 'temperature': 0, 'ignore_eos': True, 'min_tokens': gen,
        'stream_options': {'include_usage': True},
        'messages': [{'role': 'user', 'content': f'[{tag}-{i}] ' + corpus[off:off + CH]}]}).encode(), {'content-type': 'application/json'})
    t0 = time.time(); marks = []; ptok = None
    for line in urllib.request.urlopen(req, timeout=600):
        if line.startswith(b'data: {'):
            d = json.loads(line[6:])
            if d.get('usage'): ptok = d['usage'].get('prompt_tokens')
            if d.get('choices') and d['choices'][0].get('delta', {}).get('content'):
                marks.append(time.time() - t0)
    out[i] = (marks, ptok)

def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]

for k, n in enumerate([2] + conc):  # the first burst (2 streams, 32 tokens) only warms up
    out = [None] * n; tag = f'lat{k}-{time.time():.0f}'
    m0 = metrics()
    ts = [threading.Thread(target=one, args=(i, tag, out)) for i in range(n)]
    t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]; wall = time.time() - t0
    if k == 0: time.sleep(1); continue
    m1 = metrics(); dm = {a: m1[a] - m0[a] for a in m0}
    ttft = sorted(o[0][0] for o in out); gaps = [b - a for o in out for a, b in zip(o[0], o[0][1:])]
    toks = sum(len(o[0]) for o in out)
    print(f'streams {n} wall {wall:.2f}s tok {toks} ptok {sorted(set(o[1] for o in out))} | ttft med {pct(ttft, 0.5):.3f} p90 {pct(ttft, 0.9):.3f} '
          f'max {ttft[-1]:.3f} all {[round(x, 2) for x in ttft]} | gap p50 {1e3 * pct(gaps, 0.5):.1f}ms p99 {1e3 * pct(gaps, 0.99):.1f}ms '
          f'max {1e3 * max(gaps):.0f}ms >0.25s {sum(g > 0.25 for g in gaps)} >0.5s {sum(g > 0.5 for g in gaps)} of {len(gaps)} | '
          f'server: packs {dm["packs"]:.0f} chunks {dm["chunks"]:.0f} itl>0.25 {dm["n"] - dm["le025"]:.0f} >0.5 {dm["n"] - dm["le05"]:.0f} '
          f'>1 {dm["n"] - dm["le1"]:.0f}', flush=True)
    time.sleep(2)
