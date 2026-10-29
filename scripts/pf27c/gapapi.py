#!/usr/bin/env python3
# Lane pf27c trace helper (rocprofv3 kernel trace csv.gz of one tp2 generate; see docs/results/2026-10-29-pf27c-hip-27b-prefill.md)
import csv,gzip,sys
D=sys.argv[1]; pat=sys.argv[2]; ag=sys.argv[3] if len(sys.argv)>3 else 'Agent 1'
k=[r for r in csv.DictReader(gzip.open(D+'.csv.gz','rt')) if r['Agent_Id']==ag]
k.sort(key=lambda r:int(r['Start_Timestamp']))
h={r['Correlation_Id']:r for r in csv.DictReader(gzip.open(D+'.hip.csv.gz','rt'))}
hl=sorted(h.values(),key=lambda r:int(r['Start_Timestamp']))
nm=lambda r:r['Kernel_Name'].replace('(anonymous namespace)::','').split('(')[0][-40:]
us=lambda a,b:(int(a)-int(b))/1e3
n=0
for i,r in enumerate(k):
    if pat in nm(r) and us(r['Start_Timestamp'],k[i-1]['End_Timestamp'])>100:
        a=h.get(r['Correlation_Id']); p=h.get(k[i-1]['Correlation_Id'])
        print('gap %.0f | prev %s api %s->%s | this api start %.0f us after prev kernel end, dur %.0f, thread %s' % (us(r['Start_Timestamp'],k[i-1]['End_Timestamp']), nm(k[i-1]), p and p['Function'], a and a['Function'], us(a['Start_Timestamp'],k[i-1]['End_Timestamp']), us(a['End_Timestamp'],a['Start_Timestamp']), a['Thread_Id']))
        # API calls of that thread between prev launch end and this launch start
        if p:
            mid=[x for x in hl if x['Thread_Id']==a['Thread_Id'] and int(p['End_Timestamp'])<=int(x['Start_Timestamp'])<int(a['Start_Timestamp'])]
            print('    between: %d calls %s; prev launch dur %.0f us, ended %.0f us before prev kernel end' % (len(mid), sorted(set(x['Function'] for x in mid)), us(p['End_Timestamp'],p['Start_Timestamp']), us(k[i-1]['End_Timestamp'],p['End_Timestamp'])))
        n+=1
        if n>=4: break
