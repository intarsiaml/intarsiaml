import sys, json
import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
tok = Tokenizer.from_file('/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4/tokenizer.json')
f = safe_open(sys.argv[1], 'np')
i = int(sys.argv[2]); a = int(sys.argv[3]); b = int(sys.argv[4])
ids = f.get_tensor(f'ids_{i}'); tk = f.get_tensor(f'topk_ids_{i}'); tl = f.get_tensor(f'topk_logprobs_{i}').astype(np.float64)
for t in range(a, b):
    nxt = tok.decode([int(ids[t + 1])])
    tops = " | ".join("%d:%r %.2f" % (int(x), tok.id_to_token(int(x)), l) for x, l in zip(tk[t][:3], tl[t][:3]))
    print('%5d  next=%-14r  %s' % (t, nxt, tops))
