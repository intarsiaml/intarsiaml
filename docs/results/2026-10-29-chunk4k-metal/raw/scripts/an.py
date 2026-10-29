import re,sys,statistics as st,glob,os
D=sys.argv[1]
def num(s): return float(re.match(r'[\d.]+',s.strip()).group())
single={}  # (arm,size)->{'pre':[],'dec':[],'ttft':[]}
burst={}   # (arm,metric key)->list
for arm in ('c2048','c4096'):
    for r in (1,2,3):
        t=open(f'{D}/r{r}-{arm}-single-localeval.out').read()
        for l in t.split('\n'):
            m=re.match(r'\| (\d+) \| ([\d.]+) \| (\d+) \(.*?\)\s*\*? \| \S+ \| ([\d.]+) \(',l)
            if m:
                size=int(m.group(1)); size={991:'1k',4090:'4k'}.get(size,'16k' if size>10000 else '1k' if size<1100 else '4k')
                d=single.setdefault((arm,size),{'ttft':[],'pre':[],'dec':[]})
                d['ttft'].append(float(m.group(2)));d['pre'].append(float(m.group(3)));d['dec'].append(float(m.group(4)))
        t=open(f'{D}/r{r}-{arm}-burst-localeval.out').read()
        sec=t.split('## Speed sweep')[1] if '## Speed sweep' in t else t
        for l in sec.split('\n'):
            m=re.match(r'\| (\d+) \| ([\d.]+) \| ([\d.]+) \(.*?\) \| ([\d.]+) \|$',l)   # streams table
            if m: burst.setdefault((arm,int(m.group(1))),{'agg':[],'ttft':[]}); burst[(arm,int(m.group(1)))]['agg'].append(float(m.group(2))); burst[(arm,int(m.group(1)))]['ttft'].append(float(m.group(4)))
            m=re.match(r'\| (\d+) \| ([\d.]+) \| (\d+) \(.*?\)\s*\*? \| \S+ \| ([\d.]+) \(',l)   # 1-stream row
            if m and int(m.group(1))<1100:
                d=burst.setdefault((arm,1),{'agg':[],'ttft':[]}); d['agg'].append(float(m.group(4))); d['ttft'].append(float(m.group(2)))
def pct(a,b): return (b-a)/a*100
def f(v): return ' / '.join(f'{x:g}' for x in v)
print('## single stream (default serve), per-round values r1 / r2 / r3 (each = median of 3 reps), then median of rounds\n')
for size in ('1k','4k','16k'):
    for key,name in (('pre','prefill tok/s'),('dec','decode tok/s'),('ttft','TTFT s')):
        pass
rows=[]
print('| size | metric | 2048 r1/r2/r3 | 4096 r1/r2/r3 | median 2048 | median 4096 | change |')
print('|---|---|---|---|---:|---:|---:|')
for size in ('1k','4k','16k'):
    for key,name in (('pre','prefill tok/s'),('dec','decode tok/s')):
        vals={}
        for arm in ('c2048','c4096'):
            v=single[(arm,size)][key]; vals[arm]=v
        a,b=st.median(vals['c2048']),st.median(vals['c4096'])
        print(f"| {size} | {name} | {f(vals['c2048'])} | {f(vals['c4096'])} | {a:g} | {b:g} | {pct(a,b):+.1f} % |")
print('\n## burst (--max-batch 16, 1k prompts, 512 tokens), aggregate decode tok/s; 1 stream = decode of the single request\n')
print('| streams | 2048 r1/r2/r3 | 4096 r1/r2/r3 | median 2048 | median 4096 | change |')
print('|---:|---|---|---:|---:|---:|')
for n in (1,4,8,16):
    a=burst[('c2048',n)]['agg']; b=burst[('c4096',n)]['agg']
    print(f"| {n} | {f(a)} | {f(b)} | {st.median(a):g} | {st.median(b):g} | {pct(st.median(a),st.median(b)):+.1f} % |")
