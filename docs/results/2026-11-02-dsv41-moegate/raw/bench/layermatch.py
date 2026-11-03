import csv, gzip, sys, statistics as st, collections
def rows(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        r = csv.DictReader(f)
        for d in r:
            yield d
def short(n):
    n = n.split('(const')[0]
    return n
def table(path, pred, label, key):
    # collect launches in time order for the kernel `key` function over 2048-row grids
    ks = collections.defaultdict(list)
    for d in rows(path):
        n = d['Name']
        k = key(d, n)
        if k: ks[k].append(float(d['Duration (ns)'])/1e6)
    return ks
