import re,glob,sys,collections,statistics as st

import gzip


def opn(fn):
    return gzip.open(fn, "rt", errors="replace") if fn.endswith(".gz") else open(fn, errors="replace")


G=re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+)")
C=re.compile(r"^spec: copy token \d+ pos (\d+) drafted (\d+) accepted (\d+)")
def parse(fn):
    reqs=[];cur=None;last=-1
    ev=[]  # chain events as (req index, pos, drafted, accepted)
    for l in opn(fn):
        g=G.match(l)
        if g:
            pos=int(g.group(4))
            if cur is None or pos<last-50: cur={'steps':[],'ch':[]};reqs.append(cur)
            last=pos;cur['steps'].append((pos,float(g.group(1))));continue
        c=C.match(l)
        if c and cur is not None: cur['ch'].append((int(c.group(1)),int(c.group(2)),int(c.group(3))))
    return reqs[2:]
d=sys.argv[1]; B=50
rows=[]
for r in range(1,5):
    base=parse(f"{d}/serve-base-r{r}.log.gz"); arm=parse(f"{d}/serve-n20-r{r}.log.gz")
    total=0; acc=0; first_req=None
    for i,(ra,rb) in enumerate(zip(arm,base)):
        bb=collections.defaultdict(list); ab=collections.defaultdict(list)
        for p,ms in rb['steps']: bb[p//B].append(ms)
        for p,ms in ra['steps']: ab[p//B].append(ms)
        for b in sorted(ab):
            if b*B<100 or b not in bb or len(ab[b])<10 or len(bb[b])<10: continue
            cp=[c for c in ra['ch'] if c[0]<b*B]  # chains in this request before the bin
            seen=total+len(cp); accs=acc+sum(c[2] for c in cp)
            rows.append((r,i,b,seen,accs,st.mean(ab[b])-st.mean(bb[b])))
        total+=len(ra['ch']); acc+=sum(c[2] for c in ra['ch'])
by=collections.defaultdict(list)
for r,i,b,seen,accs,dd in rows:
    k='0' if seen==0 else '1' if seen==1 else '2-3' if seen<=3 else '4-6' if seen<=6 else '7+'
    by[k].append(dd)
for k in ['0','1','2-3','4-6','7+']:
    if by[k]: print(f"chains seen so far {k:4s}: mean diff {st.mean(by[k]):+.3f} ms (sd {st.pstdev(by[k]):.3f}, n {len(by[k])})")
# only chains that accepted 0 / 1
byacc=collections.defaultdict(list)
for r,i,b,seen,accs,dd in rows:
    if seen>0: byacc['accepted-so-far 0-3' if accs<=3 else 'accepted-so-far 4+'].append(dd)
for k,v in byacc.items(): print(k, f"{st.mean(v):+.3f} n={len(v)}")
# per request index within round: any drift in later requests (seen vs request index)
byreq=collections.defaultdict(list)
for r,i,b,seen,accs,dd in rows: byreq[i].append(dd)
print('by request index:', ' '.join(f"{i}:{st.mean(v):+.2f}" for i,v in sorted(byreq.items())))
