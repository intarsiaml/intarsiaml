import json, sys, statistics as st
from collections import Counter
a={r['i']:r for r in map(json.loads,open(sys.argv[1]))}
b={r['i']:r for r in map(json.loads,open(sys.argv[2]))}
ks=sorted(set(a)&set(b))
print('n',len(ks),'A',sum(a[k]['score'] for k in ks),'B',sum(b[k]['score'] for k in ks))
pm=[(k,a[k]['pt'],b[k]['pt']) for k in ks if a[k]['pt']!=b[k]['pt']]
print('pt mismatch',len(pm),pm[:10])
print('identical text',sum(a[k]['text']==b[k]['text'] for k in ks))
iw=[k for k in ks if a[k]['score']<b[k]['score']]; vw=[k for k in ks if a[k]['score']>b[k]['score']]
print('A wrong B right',len(iw),iw[:30]); print('B wrong A right',len(vw),vw[:30])
print('ct median', st.median(a[k]['ct'] for k in ks), st.median(b[k]['ct'] for k in ks))
print('long(>20)', sum(a[k]['ct']>20 for k in ks), sum(b[k]['ct']>20 for k in ks))
print('A long & B short', sum(a[k]['ct']>20 and b[k]['ct']<=20 for k in ks), 'B long & A short', sum(b[k]['ct']>20 and a[k]['ct']<=20 for k in ks))
print(Counter(a[k]['finish'] for k in ks), Counter(b[k]['finish'] for k in ks))
