import re,sys,os,statistics as st
d=sys.argv[1]
base=None
def rows(f):
    R={}
    for l in open(f,errors='replace'):
        m=re.search(r'rows=(\d+): build ([\d.]+) ms, enqueue ([\d.]+) ms, whole pass ([\d.]+) ms, engram wait ([\d.]+)',l)
        if m: R.setdefault(int(m[1]),[]).append(tuple(float(m[i]) for i in (2,3,4,5)))
    return R
order=[l.split()[2].rstrip(':') for l in open(d+'/steps.log') if re.search(r' arm \S+:( |$)',l)]
print('arm | text same as first arm | rounds | rows=3: n, whole ms median, p25, build, enqueue, engram wait | rows=2 whole | segments | warns')
for n in order:
    f=f'{d}/{n}.prof'
    if not os.path.exists(f): print(n,'no data'); continue
    txt=open(f'{d}/{n}.txt').read()
    if base is None: base=txt
    R=rows(f); t3=R.get(3,[]); t2=R.get(2,[])
    lines=open(f,errors='replace').read()
    rounds=sum(int(x) for x in re.findall(r'in (\d+) rounds',lines))
    t5=R.get(5,[]); segs=sorted(set(re.findall(r'captured the (\d+)-row verify pass as (\d+) CUDA',lines)))
    w=sorted(x[2] for x in t3)
    print(n,'|',txt==base,len(txt.splitlines()),'|',rounds,'|',len(t3),round(st.median(w),2) if w else None,round(w[len(w)//4],2) if w else None,*(round(st.median(x[i] for x in t3),3) for i in (0,1,3)) if t3 else (),'|',round(st.median(x[2] for x in t2),2) if t2 else None,'|',segs[:2],'|',lines.count('NCCL WARN'),'| draft rows=5',round(st.median(x[2] for x in t5),2) if t5 else None)
