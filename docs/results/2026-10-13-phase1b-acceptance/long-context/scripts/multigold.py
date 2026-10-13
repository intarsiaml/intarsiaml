import json,struct,sys
# usage: multigold.py src out idx:len [idx:len ...]
src,out=sys.argv[1],sys.argv[2]; specs=[tuple(int(x) for x in a.split(':')) for a in sys.argv[3:]]
f=open(src,'rb'); n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); base=8+n
def get(name):
    t=h[name]; f.seek(base+t['data_offsets'][0]); return f.read(t['data_offsets'][1]-t['data_offsets'][0])
meta=dict(h['__metadata__']); prompts=json.loads(meta['prompts'])
ps=[]; hdr={}; blob=b''
def add(k,dt,b,sh):
    global blob
    hdr[k]={'dtype':dt,'shape':sh,'data_offsets':[len(blob),len(blob)+len(b)]}; blob+=b
for j,(pi,L) in enumerate(specs):
    ids=get(f'ids_{pi}'); ti=get(f'topk_ids_{pi}'); tl=get(f'topk_logprobs_{pi}')
    N=len(ids)//4; K=len(ti)//4//N; assert L<=N
    p=dict(prompts[pi]); p['name']=f"{p['name']}@{L}"; p['n_tokens']=L; p['n_prompt']=L; p['continuation']=0; p.pop('reuse_split',None); ps.append(p)
    add(f'ids_{j}','I32',ids[:4*L],[L]); add(f'topk_ids_{j}','I32',ti[:4*K*L],[L,K]); add(f'topk_logprobs_{j}','F32',tl[:4*K*L],[L,K])
meta['prompts']=json.dumps(ps); hdr['__metadata__']=meta
hb=json.dumps(hdr).encode(); hb+=b' '*((8-len(hb)%8)%8)
open(out,'wb').write(struct.pack('<Q',len(hb))+hb+blob); print(len(specs),'prompts')
