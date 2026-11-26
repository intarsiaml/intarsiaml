import json, sys, glob
# usage: analyze_gen.py cb-*.jsonl : prefill window = first embedding cb (nd<=8) followed by a cb with nd>=97, up to the first logits_gather_p12 after it
print("file | span_ms busy_ms idle_ms idle% | lead_ms bound_ms(n,max) tail_ms other_ms | emb0 dur | hc-prev_e at boundaries (ms)")
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]; rows.sort(key=lambda r: r['s'])
    a = next(k for k in range(len(rows) - 1) if rows[k]['op'] == 'embedding_bf16' and rows[k]['nd'] <= 8 and rows[k + 1]['nd'] >= 97)
    b = next(k for k in range(a, len(rows)) if rows[k]['op'] == 'logits_gather_p12')
    p = rows[a:b + 1]
    span = (p[-1]['e'] - p[0]['s']) * 1e3; busy = sum(r['e'] - r['s'] for r in p) * 1e3
    lead = bound = tail = other = 0.0; nb = 0; mb = 0.0; seen = False; hcs = []
    for k in range(1, len(p)):
        g = (p[k]['s'] - p[k - 1]['e']) * 1e3; big = p[k]['nd'] >= 97
        if p[k]['op'] == 'embedding_bf16' and p[k]['nd'] <= 8 and seen:
            hcs.append((p[k]['hc'] - p[k - 1]['e']) * 1e3)
        if g > 0:
            if p[k]['op'] == 'embedding_bf16' and p[k]['nd'] <= 8 and seen: bound += g; nb += 1; mb = max(mb, g)
            elif big and not seen: lead += g
            elif p[k]['op'] == 'logits_gather_p12': tail += g
            else: other += g
        seen = seen or big
    print(f"{f.split('/')[-1]} | {span:8.1f} {busy:8.1f} {span-busy:7.1f} {100*(span-busy)/span:5.2f} | {lead:6.1f} {bound:6.1f}({nb},{mb:.1f}) {tail:6.1f} {other:6.1f} | {(p[0]['e']-p[0]['s'])*1e3:.1f} | {' '.join('%.1f' % h for h in hcs)}")
