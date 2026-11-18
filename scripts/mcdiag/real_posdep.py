import re,collections,statistics as st,sys

import gzip


def opn(fn):
    return gzip.open(fn, "rt", errors="replace") if fn.endswith(".gz") else open(fn, errors="replace")


sys.path.insert(0,'/private/tmp/claude-501/-opt-code-intarsiaml/a9e6e738-26f8-4e34-872e-c7d989747c74/scratchpad')
G=re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+)")
C=re.compile(r"^spec: copy token \d+ pos (\d+)")
def parse(fn):
    reqs=[];cur=None;last=-1
    for l in opn(fn):
        g=G.match(l)
        if g:
            pos=int(g.group(4))
            if cur is None or pos<last-50: cur={'steps':[],'ch':[]};reqs.append(cur)
            last=pos;cur['steps'].append((pos,float(g.group(1)),int(g.group(3))));continue
        c=C.match(l)
        if c and cur is not None: cur['ch'].append(int(c.group(1)))
    return reqs[2:]
d=sys.argv[1];B=100
res=collections.defaultdict(list);resn=collections.defaultdict(list);byn=collections.defaultdict(list)
for r in range(1,5):
    base=parse(f"{d}/serve-base-r{r}.log.gz");arm=parse(f"{d}/serve-n20-r{r}.log.gz");ctl=parse(f"{d}/serve-n256-r{r}.log.gz")
    total=0
    for ra,rb,rc in zip(arm,base,ctl):
        bb=collections.defaultdict(list);ab=collections.defaultdict(list);cb=collections.defaultdict(list)
        for p,ms,n in rb['steps']: bb[p//B].append(ms)
        for p,ms,n in ra['steps']: ab[p//B].append(ms)
        for p,ms,n in rc['steps']: cb[p//B].append(ms)
        for b in ab:
            if b*B<100 or b not in bb or len(ab[b])<20 or len(bb[b])<20: continue
            seen = total>0 and True
            key=('seen' if total>0 else 'unseen')
            res[(key,b*B)].append(st.mean(ab[b])-st.mean(bb[b]))
            if b in cb and len(cb[b])>=20: resn[('ctl',b*B)].append(st.mean(cb[b])-st.mean(bb[b]))
        total+=len(ra['ch'])
for k in ['unseen','seen']:
    print(k,' '.join(f"{p}:{st.mean(v):+.2f}({len(v)})" for (kk,p),v in sorted(res.items()) if kk==k))
print('n256 ',' '.join(f"{p}:{st.mean(v):+.2f}({len(v)})" for (kk,p),v in sorted(resn.items())))
