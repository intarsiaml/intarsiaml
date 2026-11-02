import csv,re,collections,sys,statistics as st
TF='out/tf-r0_cuda_gpu_trace.csv'
import os
OURS=os.environ.get('OURS_CSV','/opt/code/intarsiaml/.worktrees/dsv41-graphs/.scratch/full-w9/ab1/nsys-r0-033722_cuda_gpu_trace.csv')
def load(p):
    R=[]
    for d in csv.DictReader(open(p)):
        if d['Name'].startswith('[CUDA'): continue
        R.append(dict(t=int(d['Start (ns)']),d=int(d['Duration (ns)']),n=d['Name'],gx=d['GrdX'],gy=d['GrdY'],gz=d['GrdZ'],bx=d['BlkX'],s=d['Strm']))
    R.sort(key=lambda x:x['t']); return R
def short(n):
    n=n.replace('void ','').replace('(anonymous namespace)::','').replace('<unnamed>::','')
    m=re.match(r'([\w:]+)(<[^(]*>)?',n)
    return (m.group(1).split('::')[-1]+(m.group(2) or '')) [:44]
def tf3():
    r=load(TF); s=[i for i,x in enumerate(r) if x['n'].startswith('_embed_init')]
    return [r[a:b] for a,b in zip(s,s[1:]+[len(r)]) if r[a]['gx']=='3']
def ours3():
    r=load(OURS); ps=[];c=[]
    for x in r:
        c.append(x)
        if 'AllGather' in x['n']: ps.append(c);c=[]
    out=[]
    for i in range(len(ps)-1):
        if len(ps[i])>1000 and len(ps[i+1])<400:
            am=[x for x in ps[i+1][:4] if 'argmax' in x['n']]
            if am and am[0]['gx']==os.environ.get('VROWS','3'): out.append(ps[i])
    return out
if __name__=='__main__':
  for label,ps in (('TF',tf3()),('ours',ours3())):
    n=len(ps); A=collections.defaultdict(lambda:[0,0])
    for p in ps:
        for x in p:
            k=(short(x['n']),x['gx'],x['gy'],x['gz'],x['bx'])
            A[k][0]+=1/n; A[k][1]+=x['d']/n/1e6
    print('=====',label,n,'passes; kernels/pass',sum(v[0] for v in A.values()))
    agg=collections.defaultdict(lambda:[0,0])
    for k,v in A.items(): agg[k[0]][0]+=v[0]; agg[k[0]][1]+=v[1]
    for k,v in sorted(agg.items(),key=lambda kv:-kv[1][1])[:40]: print(f'{k:46s} {v[0]:7.1f}x {v[1]:7.3f} ms {1000*v[1]/v[0]:7.1f} us')
