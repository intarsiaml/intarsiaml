#!/usr/bin/env python3
import json,re,sys
reqs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
done=[l for l in open(sys.argv[2]) if 'done' in l]
done=done[-len(reqs):] if reqs else []
tt=rr=0; ds=0.0; ttft=[]; acc=[0,0]
rows=[]
for q,d in zip(reqs,done):
    m=dict(re.findall(r'(\w+)=(\S+)',d))
    tok=int(m['tokens']); rounds=int(m['rounds']); a,b=map(int,m['accepted'].split('/'))
    ds_=q['decode_s'] or 0.0
    rows.append((q['prompt'][-11:-4],tok,rounds,ds_,q['ttft_s']))
    tt+=tok-1; rr+=rounds; ds+=ds_; ttft.append(q['ttft_s'] or 0); acc[0]+=a; acc[1]+=b
for p,tok,rounds,d,t in rows:
    print(f'  {p} tokens {tok:4d} rounds {rounds:3d} decode {d:6.2f}s  {1000*d/max(rounds,1):6.2f} ms/round  {(tok-1)/max(d,1e-9):6.1f} tok/s  ttft {t:.2f}s')
if rr: print(f'  TOTAL {len(rows)} requests: {tt} decode tokens in {rr} rounds = {tt/rr:.2f} tok/round, accepted {acc[0]}/{acc[1]}; {1000*ds/rr:.2f} ms/round, {tt/ds:.1f} tok/s decode; mean ttft {sum(ttft)/len(ttft):.2f}s')
