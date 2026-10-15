import sys, json
# usage: cb_an.py PROFILE.jsonl : per-prefill (4k prompt = 15 chunks of t=256 then a remainder chunk, 3 CBs per chunk):
# wall span first-start..last-end, GPU busy sum, idle gaps between command buffers, per-CB mean duration.
recs = [json.loads(l) for l in open(sys.argv[1])]
pre = [r for r in recs if r['t'] > 100]
groups = []; cur = []; seen_rem = False
for r in pre:
    if r['t'] == 256 and seen_rem:
        groups.append(cur); cur = []; seen_rem = False
    if r['t'] != 256: seen_rem = True
    cur.append(r)
if cur: groups.append(cur)
print("grp n_cb span_ms gpu_busy_ms idle_ms (idle%) maxgap_ms busy/cb_ms  first4chunks_span_ms")
k = 0
for g in groups:
    if len(g) < 40: continue
    span = (g[-1]['e'] - g[0]['s']) * 1e3
    busy = sum(r['e'] - r['s'] for r in g) * 1e3
    gaps = [(g[i]['s'] - g[i - 1]['e']) * 1e3 for i in range(1, len(g))]
    idle = sum(x for x in gaps if x > 0)
    early = (g[11]['e'] - g[0]['s']) * 1e3
    print("%3d %4d %7.0f %7.0f %6.0f (%4.1f%%) %6.1f %5.1f %6.0f" % (k, len(g), span, busy, idle, 100 * idle / span, max(gaps), busy / len(g), early))
    k += 1
