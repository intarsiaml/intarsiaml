#!/usr/bin/env python3
# Lane pf27c trace helper (rocprofv3 kernel trace csv.gz of one tp2 generate; see docs/results/2026-10-29-pf27c-hip-27b-prefill.md)
import csv,gzip,collections,sys
rows=[r for r in csv.DictReader(gzip.open(sys.argv[1],'rt')) if r['Agent_Id']=='Agent 1']
rows.sort(key=lambda r:int(r['Start_Timestamp']))
st=[i for i,r in enumerate(rows) if 'mx_a8_quant' in r['Kernel_Name']][0]; rows=rows[st-5:]
nm=lambda r:r['Kernel_Name'].replace('(anonymous namespace)::','').split('(')[0][:60]
g=collections.defaultdict(lambda:[0,0.0])
for p,r in zip(rows,rows[1:]):
    gap=max(0,(int(r['Start_Timestamp'])-int(p['End_Timestamp']))/1e3)
    k=nm(p)+' -> '+nm(r); g[k][0]+=1; g[k][1]+=gap
print('total %.1f ms, %d kernels'%(sum(v[1] for v in g.values())/1e3,len(rows)))
for k,v in sorted(g.items(),key=lambda kv:-kv[1][1])[:10]: print('%6.2f ms %4d %6.1f us  %s'%(v[1]/1e3,v[0],v[1]/v[0],k))
