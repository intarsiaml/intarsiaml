import json, sys
# usage: analyze.py cb.jsonl requests.tsv
rows = [json.loads(l) for l in open(sys.argv[1])]
rows.sort(key=lambda r: r['s'])
print("req size idle  prompt cached | client_s | gpu_span_ms busy_ms idle_ms idle% | lead_ms bound_ms(n,max) tail_ms other_ms | s-hc first_cb")
tot = {}
for line in open(sys.argv[2]):
    n, sz, idle, s, e, t, pt, ct = line.split()
    s = float(s); e = float(e)
    w = [r for r in rows if r['s'] >= s - 0.02 and r['e'] <= e + 0.02]
    if not w: print(n, 'no cbs'); continue
    # prefill region: first cb .. last cb that belongs to the chunks (stop at the cb with op logits_gather_p12, the end of prefill)
    end = max([k for k, r in enumerate(w) if r['op'] == 'logits_gather_p12'] or [len(w) - 1])
    p = w[:end + 1]
    span = (p[-1]['e'] - p[0]['s']) * 1e3
    busy = sum((r['e'] - r['s']) for r in p) * 1e3
    idle_total = span - busy
    lead = bound = tail = other = 0.0; nb = 0; mb = 0.0
    seen_big = False
    for k in range(1, len(p)):
        g = (p[k]['s'] - p[k - 1]['e']) * 1e3
        big = p[k]['nd'] >= 97
        if g > 0:
            if p[k]['op'] == 'embedding_bf16' and p[k]['nd'] <= 8 and seen_big:
                bound += g; nb += 1; mb = max(mb, g)
            elif big and not seen_big:
                lead += g
            elif p[k]['op'] == 'logits_gather_p12':
                tail += g
            else:
                other += g
        seen_big = seen_big or big
    # first big cb (any non-embedding with nd>=97)
    fb = next((r for r in p if r['nd'] >= 97), p[0])
    print(f"{n:>3} {sz:>4} {idle:>3} {pt:>7} {ct:>6} | {float(t):7.3f} | {span:9.1f} {busy:8.1f} {idle_total:7.1f} {100*idle_total/span:5.2f} | {lead:6.1f} {bound:6.1f}({nb},{mb:.1f}) {tail:6.1f} {other:6.1f} | {1e3*(fb['s']-fb['hc']):.1f}")
