# usage: python3 an6.py DIR   (DIR holds r{1..6}-{c2048,c4096}-single-localeval.out); prints per-size medians and paired changes
import re,statistics as st,sys
D=sys.argv[1]
def rows(arm,r):
    out={}
    for l in open(f'{D}/r{r}-{arm}-single-localeval.out'):
        m=re.match(r'\| (\d+) \| ([\d.]+) \| (\d+) \(.*?\)\s*\*? \| \S+ \| ([\d.]+) \(',l)
        if m:
            n=int(m.group(1)); k='1k' if n<1100 else '4k' if n<5000 else '16k'
            out[k]=(float(m.group(3)),float(m.group(4)))
    return out
for rs,name in (((1,2,3),'planned 3 rounds'),((4,5,6),'extra 3 rounds'),((1,2,3,4,5,6),'all 6')):
    print(name)
    for k in ('1k','4k','16k'):
        for i,met in ((0,'prefill'),(1,'decode')):
            a=[rows('c2048',r)[k][i] for r in rs]; b=[rows('c4096',r)[k][i] for r in rs]
            pairs=[(y-x)/x*100 for x,y in zip(a,b)]
            print(f' {k} {met}: 2048 {a} 4096 {b} median {st.median(a):g} -> {st.median(b):g} ({(st.median(b)-st.median(a))/st.median(a)*100:+.1f}%) paired {[round(p,1) for p in pairs]}')
