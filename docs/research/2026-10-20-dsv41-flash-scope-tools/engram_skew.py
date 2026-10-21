import sys, os, random, json, time, collections
import numpy as np
from tokenizers import Tokenizer, Regex, normalizers
from sympy import isprime
# usage: uv run --with tokenizers --with numpy --with sympy python engram_skew.py TOKENS TOKENIZER_JSON CORPUS_ROOT
# TOKENIZER_JSON = https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/tokenizer.json
RAW = sys.argv[2] if len(sys.argv) > 2 else "tokenizer.json"
tok = Tokenizer.from_file(RAW)
n = tok.get_vocab_size(with_added_tokens=True)
sentinel = ""
normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
  normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(), normalizers.Replace(sentinel, " ")])
key_to_new, lookup = {}, [0]*n
for i in range(n):
    text = tok.decode([i], skip_special_tokens=False)
    if "�" in text: key = tok.id_to_token(i)
    else:
        nn = normalizer.normalize_str(text); key = nn if nn else text
    j = key_to_new.get(key)
    if j is None: j = len(key_to_new); key_to_new[key] = j
    lookup[i] = j
V = len(key_to_new); print("vocab", n, "compressed", V, "(expect 99092)")
assert V == 99092
token_map = np.array(lookup, dtype=np.int64); pad = token_map[2]
# layout
layer_ids=[1,14]; maxn=4; heads=8; ev=16_000_000
seen=set(); primes=[]
def nxt(s):
    c=s+1
    while not isprime(c) or c in seen: c+=1
    return c
for _ in layer_ids:
    per=[]
    for _ in range(maxn-1):
        cur=ev-1; sizes=[]
        for _ in range(heads):
            cur=nxt(cur); seen.add(cur); sizes.append(cur)
        per.append(sizes)
    primes.append(per)
print("rows/layer", [sum(sum(p) for p in per) for per in primes], "(expect 384006168, 384016682)")
bound = max(1,(np.iinfo(np.int64).max//V)//2)
mult = np.stack([np.random.default_rng(10007*l).integers(0,bound,size=(maxn,),dtype=np.int64)*2+1 for l in layer_ids])
offs=[]
for per in primes:
    flat=[p for x in per for p in x]; offs.append(np.cumsum([0,*flat[:-1]]))
def hashes(ids):  # ids: compressed int64 [L] -> [L, nlayers, 24] row ids
    L=len(ids); toks=[]
    for s in range(maxn):
        src=np.concatenate([np.full(s,pad,dtype=np.int64), ids[:L-s]]) if s else ids
        toks.append(src)
    toks=np.stack(toks,-1)  # [L,4]  (positions<shift padded, pad is the same as blocked->pad)
    out=np.zeros((L,len(layer_ids),(maxn-1)*heads),dtype=np.int64)
    for li in range(len(layer_ids)):
        prod=toks*mult[li]; rolling=prod[:,0].copy(); cols=[]
        for i in range(1,maxn):
            rolling=np.bitwise_xor(rolling,prod[:,i])
            cols.append(rolling[:,None] % np.array(primes[li][i-1],dtype=np.int64)[None,:])
        out[:,li,:]=np.concatenate(cols,-1)+offs[li]
    return out
# corpus
random.seed(0)
files=[]
for root,dirs,fs in os.walk(sys.argv[3] if len(sys.argv) > 3 else "."):
    dirs[:]=[d for d in dirs if d not in ("node_modules","target",".venv","venv",".git","dist","build","__pycache__","site-packages",".cache")]
    for f in fs:
        if f.endswith((".md",".py",".rs",".txt")):
            p=os.path.join(root,f)
            try:
                if 2000<os.path.getsize(p)<400_000: files.append(p)
            except OSError: pass
random.shuffle(files); print("candidate files",len(files))
budget=int(sys.argv[1]) if len(sys.argv)>1 else 3_000_000
docs=[]; total=0
for p in files:
    try: t=open(p,encoding="utf-8",errors="ignore").read()
    except: continue
    ids=np.array(tok.encode(t).ids,dtype=np.int64)
    if len(ids)<64: continue
    docs.append(token_map[ids]); total+=len(ids)
    if total>=budget: break
print("docs",len(docs),"tokens",total)
H=[hashes(d) for d in docs]
split=len(H)//2
A=np.concatenate([h.reshape(-1) for h in H[:split]]); B=np.concatenate([h.reshape(-1) for h in H[split:]])
print("lookups train",len(A),"test",len(B), "tokens/lookups per token", 48)
# hot-set: pick top-K rows by train frequency (global, across all 2*24 tables), evaluate on test
ua,ca=np.unique(A,return_counts=True); order=np.argsort(-ca); ua=ua[order]
print("distinct rows train",len(ua),"of",len(A),"lookups")
ub,cb=np.unique(B,return_counts=True)
print("distinct rows test",len(ub), "-> within-test unique/lookup =",len(ub)/len(B))
res={}
for K in (10_000,100_000,1_000_000,4_000_000):
    hot=ua[:K]; m=np.isin(B,hot).mean(); res[K]=m
    print(f"hot set top-{K:>9,} rows (train) -> test coverage {m:6.1%}   cache size {K*264/1e6:8.1f} MB")
# in-sample self-coverage upper bound for Zipf shape: share of lookups in top-K rows of test itself
cs=np.cumsum(np.sort(cb)[::-1]); 
for K in (10_000,100_000,1_000_000):
    print(f"oracle(in-sample) top-{K:,}: {cs[min(K,len(cs))-1]/cs[-1]:.1%}")
# per-order split (which n-gram order is heavy tail): coverage by column group using train top 1M
hot=set(ua[:1_000_000].tolist())
Bc=np.concatenate([h for h in H[split:]],0)  # [L,2,24]
hotarr=ua[:1_000_000]
for li in range(2):
    for oi in range(3):
        sub=Bc[:,li,oi*8:(oi+1)*8].reshape(-1)
        print(f"layer idx {li} order {oi+2}-gram: top-1M(train) coverage {np.isin(sub,hotarr).mean():6.1%}  unique/lookups(test) {len(np.unique(sub))/len(sub):.3f}")
# LRU sim over concatenated test docs, cache sizes in rows, stream order = token order, lookups per token 48
def lru(sizeR, maxtok=150_000):
    od=collections.OrderedDict(); hit=0; tot=0
    seq=np.concatenate(H[split:],0)[:maxtok].reshape(-1,48)
    for row in seq:
        for r in row.tolist():
            tot+=1
            if r in od: od.move_to_end(r); hit+=1
            else:
                od[r]=1
                if len(od)>sizeR: od.popitem(last=False)
    return hit/tot
for sz in (64_000,512_000,2_000_000):
    t=time.time(); print(f"LRU {sz:>9,} rows ({sz*264/1e6:7.1f} MB) hit {lru(sz):6.1%}  ({time.time()-t:.0f}s)")
