# usage: python3 -I byte_budget.py exl3.json base.json   (json from hf_headers.py)
import json,re,sys,collections
def grp(n):
    if re.search(r'engram\.embed\.(weight|scale)$',n): return '1 engram tables (embed+scale)'
    if 'engram' in n: return '2 engram resident (wkv,q,k)'
    if n.startswith('mtp.'): return '9 MTP/dspark draft'
    if n.startswith(('vision','aligner','image_')): return 'a vision tower + aligner'
    if re.search(r'\.ffn\.experts\.',n): return '3 routed experts'
    if 'shared_experts' in n: return '4 shared expert'
    if re.search(r'\.attn\.indexer\.',n): return '6 indexer'
    if re.search(r'\.hc_',n): return '7 hyper-connection'
    if re.search(r'\.attn\.',n) : return '5 attention (+compressor)'
    if re.search(r'\.ffn\.gate\.',n): return '8 router gate'
    if n.startswith('embed'): return 'b embed'
    if n.startswith('head'): return 'b head'
    return 'c norms/other'
for fn in sys.argv[1:]:
    f=json.load(open(fn)); G=collections.defaultdict(lambda:[0,0,collections.Counter()])
    for n,v in f.items():
        g=grp(n); G[g][0]+=v['nbytes']; G[g][1]+=1; G[g][2][v['dtype']]+=v['nbytes']
    tot=sum(g[0] for g in G.values()); print(fn,tot/1e9)
    for g in sorted(G): b,c,d=G[g]; print(f"{g:34s}{b/1e9:9.3f} GB {100*b/tot:5.1f}%  {c:7d}  {dict((k,round(x/1e9,3)) for k,x in d.items() if x>1e6)}")
