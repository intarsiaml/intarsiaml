import json,sys,numpy as np
sys.path.insert(0,'scripts/ac')
from noise import load,perpos
from turns import scored
g='golden/qwen3.8-27b-mxfp4-hfquark-'
files={'bf16':g+'top20.safetensors','f32':g+'f32-top20.safetensors','w4a16':g+'w4a16-top20.safetensors'}
G={k:load(v) for k,v in files.items()}
meta=G['bf16'][1]; names=[p['name'] for p in json.loads(meta['prompts'])]
def noise(a,b,K):
    tot=[];sc=[]
    for i,nm in enumerate(names):
        ids=list(G[a][0][f'ids_{i}']); k=perpos(G[a][0],G[b][0],i); m=scored(ids)
        lo=len(ids)-K if K else 0
        for p in range(lo,len(ids)):
            tot.append(k[p]); 
            if m[p]: sc.append(k[p])
    return np.mean(tot),len(tot),np.mean(sc),len(sc),max(sc)
for a,b in (('bf16','f32'),('bf16','w4a16'),('f32','w4a16')):
    for K in (0,16):
        r=noise(a,b,K); print(f'{a} vs {b} {"window16" if K else "all rows"}: all={r[0]:.4f} (n={r[1]}) scored={r[2]:.4f} (n={r[3]}) maxscored={r[4]:.2f}')
