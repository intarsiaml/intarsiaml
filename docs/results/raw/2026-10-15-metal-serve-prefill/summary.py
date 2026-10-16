import sys, collections, statistics as st
# usage: summary.py LABEL... : per-run, per-condition server-side prefill ms (n, min, median, max) and TTFT-prefill overhead
for lab in sys.argv[1:]:
    rows = [l.split('\t') for l in open(lab + '.tsv').read().splitlines()[1:]]
    if rows and rows[0][1] in ('plain', 'serve_idle5', 'serve_b2b'):
        d = collections.defaultdict(list); o = collections.defaultdict(list)
        for r in rows:
            d[r[1]].append(int(r[2]))
            if r[3]: o[r[1]].append(int(r[3]) - int(r[2]))
    else:
        d = collections.defaultdict(list); o = collections.defaultdict(list)
        for r in rows:
            d[r[1]].append(int(r[4])); o[r[1]].append(int(r[3]) - int(r[4]))
    print("== " + lab)
    for k in sorted(d):
        v = sorted(d[k])
        ov = o.get(k) or [0]
        print("  %-18s n=%2d min=%4d med=%4d max=%4d  (>1.3x floor: %d/%d)  ttft-minus-server-prefill median %d ms" % (k, len(v), v[0], st.median(v), v[-1], sum(1 for x in v if x > 1.3 * 1150), len(v), st.median(ov)))
