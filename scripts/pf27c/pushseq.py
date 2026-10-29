#!/usr/bin/env python3
# Lane pf27c trace helper (rocprofv3 kernel trace csv.gz of one tp2 generate; see docs/results/2026-10-29-pf27c-hip-27b-prefill.md)
import csv,gzip,collections,statistics as s,sys
rows=[r for r in csv.DictReader(gzip.open(sys.argv[1],'rt')) if r['Agent_Id']==sys.argv[2]]
rows.sort(key=lambda r:int(r['Start_Timestamp']))
nm=lambda r:r['Kernel_Name'].replace('(anonymous namespace)::','').split('(')[0]
dur=lambda r:(int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3
seq=[]
for i,r in enumerate(rows):
    n=nm(r)
    if n.startswith('void a8_direct_kernel<true') and n.endswith('true>'):
        j=i-1
        while not any(t in nm(rows[j]) for t in ('silu','gated_rms','flash')): j-=1
        prev=nm(rows[j])[5:20]
        k=i+1
        while 'ar2p' not in nm(rows[k]): k+=1
        # the non-push GEMM time of the same shape: not available; print the producer kernel
        seq.append((prev,dur(r),dur(rows[k]),int(r['Grid_Size_X'])//512))
c=collections.defaultdict(list)
for x in seq: c[(x[0])].append(x)
for p,v in c.items():
    print(p, len(v), 'gemm med %.0f us [%.0f-%.0f]'%(s.median([x[1] for x in v]),min(x[1] for x in v),max(x[1] for x in v)), 'ar med %.0f'%s.median([x[2] for x in v]), 'sum gemm %.1f ms ar %.1f'%(sum(x[1] for x in v)/1e3,sum(x[2] for x in v)/1e3))
