#!/usr/bin/env python3
"""usage: analyze.py DIR TAGS... -> per-arm medians from localeval result.json (filler) and nat.json (natural-v1)"""
import json,re,sys,statistics as st
d=sys.argv[1]; tags=sys.argv[2:]
def med(x):
    x=[v for v in x if v is not None]; return st.median(x) if x else None
out={}
for t in tags:
    rid=re.search(r'results/(\S+)/result.json',open(f'{d}/{t}-localeval.out').read()).group(1)
    j=json.load(open(f'/opt/code/deepeval/results/{rid}/result.json'))
    rows=j['speed']['rows'] if isinstance(j['speed'],dict) and 'rows' in j['speed'] else j['speed']
    o={'rid':rid}
    for r in rows:
        s=r['target_prompt_tokens']; g=lambda k:(r[k]['median'] if isinstance(r[k],dict) else r[k])
        o[('filler',s)]={'ptok':r['prompt_tokens'],'ttft':g('ttft_s'),'pre':g('prefill_tok_s'),'dec':g('decode_tok_s')}
    n=json.load(open(f'{d}/{t}-nat.json'))
    for s in sorted({x['size'] for x in n}):
        m=[x for x in n if x['size']==s and not x['warmup']]
        o[('nat',s)]={'ptok':m[0]['prompt_tokens'],'ttft':med([x['ttft_s'] for x in m]),'pre':med([x['prefill_tok_s'] for x in m]),'dec':med([x['decode_tok_s'] for x in m]),'decs':[round(x['decode_tok_s'],1) for x in m],'pres':[round(x['prefill_tok_s']) for x in m]}
    fp=[int(re.search(r'Footprint: (\d+) (GB|MB)',l).group(1))*(1 if 'GB' in l else 0.001) for l in open(f'{d}/{t}-footprint.txt') if 'Footprint:' in l]
    o['fp_gb']=max(fp) if fp else None
    tm=open(f'{d}/{t}-time.txt').read()
    o['peak_footprint_gb']=int(re.search(r'(\d+)\s+peak memory footprint',tm).group(1))/1e9
    o['max_rss_gb']=int(re.search(r'(\d+)\s+maximum resident set size',tm).group(1))/1e9
    out[t]=o
def show(k,f):
    print(k, ' | '.join(f"{t}: "+(f(out[t][k]) if k in out[t] else '-') for t in tags))
for k in sorted({k for o in out.values() for k in o if isinstance(k,tuple)}):
    show(k,lambda v:f"ptok {v['ptok']} ttft {v['ttft']:.3f} pre {v['pre']:.0f} dec {v['dec']:.1f}")
for t in tags: print(t,out[t]['rid'],'peak_footprint %.2f GB, max_rss %.2f GB, sampled footprint max %s GB'%(out[t]['peak_footprint_gb'],out[t]['max_rss_gb'],out[t]['fp_gb']))
json.dump({t:{str(k):v for k,v in o.items()} for t,o in out.items()},open(f'{d}/summary.json','w'),indent=1)
