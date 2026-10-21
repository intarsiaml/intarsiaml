"""How much does the per-row KL between two runs concentrate on rows where P(im_end) moves? vLLM A vs B (and gold vs CUDA)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split import load, kl21_rows, IM_END
D = os.path.dirname(os.path.abspath(__file__)) + '/'
pairs = [(D + 'rq/long-A.safetensors', D + 'rq/long-B.safetensors'), ('/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors', D + 'dumps/cuda-w3-long.dump.safetensors')]
for r, c in pairs:
    ids, rid, rlp = load(r)['chat_reuse_4k']; _, cid, clp = load(c)['chat_reuse_4k']
    kl, pir, pic, _ = kl21_rows(rid, rlp, cid[:, :20], clp[:, :20])
    u = slice(27, 4516)
    k = kl[u]; a = pir[u]; b = pic[u]
    d = np.abs(np.log(np.clip(a, 1e-6, 1)) - np.log(np.clip(b, 1e-6, 1)))
    swing = d > np.log(10)   # P(im_end) differs by more than 10x
    mid = (np.maximum(a, b) > 0.05)
    order = np.argsort(-k); top = order[: len(k) // 20]
    print(os.path.basename(r), 'vs', os.path.basename(c))
    print(f'  user rows {len(k)}  mean KL {k.mean():.3e}; rows where P(im_end) >0.05 in either: {mid.mean():.2%}, they carry {k[mid].sum() / k.sum():.1%} of KL')
    print(f'  rows where P(im_end) moves >10x (and >0.05 in one): {(swing & mid).mean():.2%}, carry {k[swing & mid].sum() / k.sum():.1%} of KL')
    print(f'  top-5% rows carry {k[top].sum() / k.sum():.1%} of KL; of them {np.mean(mid[top]):.1%} have P(im_end)>0.05 in either run')
