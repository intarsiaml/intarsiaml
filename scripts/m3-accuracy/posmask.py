"""Tool-exact per-row KL21 (INTARSIA_ACC_POS_DUMP tsv) aggregated with and without system/user-turn rows of chat prompts."""
import sys, json
import numpy as np
from safetensors import safe_open
IM_END, IM_START, ASSIST = 248046, 248045, 74455
f = safe_open('/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors', 'np')
prompts = json.loads(f.metadata()['prompts'])
masks = {}
for i, p in enumerate(prompts):
    ids = f.get_tensor(f'ids_{i}'); n = len(ids)
    sp = [t for t in range(n) if ids[t] in (IM_START, IM_END)]
    u = np.zeros(n, bool)
    for a, b in zip(sp[0::2], sp[1::2]):
        if not (a + 1 < n and ids[a + 1] == ASSIST):
            u[a:b] = True
    masks[p['name']] = u
rows = [l.rstrip('\n').split('\t') for l in open(sys.argv[1])]
rows = [r for r in rows if r[0] == 'prefill']
tot = cnt = totm = cntm = 0
per = {}
for r in rows:
    name, pos, kl = r[1], int(r[2]), float(r[5])
    tot += kl; cnt += 1
    if not masks[name][pos]:
        totm += kl; cntm += 1
    per.setdefault(name, []).append(kl)
print(f'all rows: n={cnt} mean KL21 {tot / cnt:.4e}   excluding system/user-turn rows: n={cntm} mean {totm / cntm:.4e}')
print(f'without chat_reuse_4k entirely: {(tot - sum(per["chat_reuse_4k"])) / (cnt - len(per["chat_reuse_4k"])):.4e}')
for k, v in per.items():
    print(f'  {k:14s} {np.mean(v):.4e}')
