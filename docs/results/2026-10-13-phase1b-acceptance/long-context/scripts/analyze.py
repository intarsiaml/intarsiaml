#!/usr/bin/env python3
"""Offline analysis of the chat_reuse_4k noise (section 9). Needs numpy, selfkl.py next to it, and these files:
  OUT/self_flash_{64,256,1024}.safetensors   35B self-goldens (top-64 rows) written by exp1.sh (chunk size in the name)
  OUT/w_flash{256,1024}.safetensors          written by exp3.sh (wrapper experiment)
  GOLDEN                                     golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
usage: analyze.py OUT GOLDEN
KL21 here is recomputed from top-k rows (reference top-20 ids + an "other" bucket; ids missing from the second file's
top-64 get exp(its 64th log-prob - 0.7)), so it tracks the harness number but is not identical to it."""
import json, struct, sys
import numpy as np
from selfkl import load, kl21

OUT, GOLDEN = sys.argv[1], sys.argv[2]
SPECIAL = {248044, 248045, 248046}  # <|endoftext|>, <|im_start|>, <|im_end|>


def loadg(p, pi):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
    def get(k):
        t = h[k]; f.seek(base + t['data_offsets'][0])
        a = np.frombuffer(f.read(t['data_offsets'][1] - t['data_offsets'][0]), dtype=np.int32 if t['dtype'] == 'I32' else np.float32)
        return a.reshape(t['shape'])
    return get(f'ids_{pi}'), get(f'topk_ids_{pi}'), get(f'topk_logprobs_{pi}')


def p_special(X):
    return np.array([sum(np.exp(l) for i, l in zip(X[1][t], X[2][t]) if int(i) in SPECIAL) for t in range(len(X[0]))])


def p_next(X, ids):
    out = np.zeros(len(ids) - 1)
    for t in range(len(ids) - 1):
        m = X[1][t] == ids[t + 1]
        out[t] = np.exp(X[2][t][m][0]) if m.any() else 0.0
    return out


n = 'chat_reuse_4k@4631'
V = loadg(GOLDEN, 2)
R = {k: load(f'{OUT}/self_flash_{k}.safetensors')[n] for k in (256, 64, 1024)}
print('== chunk size only (same flash kernels), mean KL21 / positions with KL21 > 1')
for a, b in ((256, 64), (256, 1024), (64, 1024)):
    for nm in (n, 'long_docs_4k@4064', 'long_code_4k@4064'):
        k = kl21(load(f'{OUT}/self_flash_{a}.safetensors')[nm], load(f'{OUT}/self_flash_{b}.safetensors')[nm])
        print(f'  chunk {a:>4} vs {b:>4}  {nm:22s} mean {k.mean():.4f}  n>1 {int((k > 1).sum())}')

print('== who is right where the two disagree sharply (p of the true next token)')
P = {'vllm': p_next(V, V[0]), **{f'c{k}': p_next(r, V[0]) for k, r in R.items()}}
print('  vLLM right (p>0.9) / ours c256 wrong (p<0.05):', int(((P['vllm'] > .9) & (P['c256'] < .05)).sum()),
      '  ours right / vLLM wrong:', int(((P['c256'] > .9) & (P['vllm'] < .05)).sum()))

print('== turn-control tokens (<|im_end|>, <|im_start|>, <|endoftext|>)')
PS = {'vllm': p_special(V), **{f'c{k}': p_special(r) for k, r in R.items()}}
amb = np.max([PS[k] for k in PS], axis=0) > 0.1
print(f'  positions where some engine puts >10% on them: {int(amb.sum())} of {len(amb)}')
for nm, (A, B) in {'ours c256 vs vLLM': (V, R[256]), 'ours c256 vs c64': (R[256], R[64]), 'ours c256 vs c1024': (R[256], R[1024])}.items():
    k = kl21(A, B)
    print(f'  {nm:20s} mean {k.mean():.4f} | on those positions {k[amb].mean():.3f} (share of all KL {k[amb].sum() / k.sum():.0%}) | elsewhere {k[~amb].mean():.4f}, n>1 {int((k[~amb] > 1).sum())}')

print('== by segment of the prompt (mean KL21)')
segs = [('system turn 0-24', 0, 25), ('user document 25-4516', 25, 4517), ('question + assistant prompt 4517-4566', 4517, 4567), ('generated <think> continuation 4567-4630', 4567, 4631)]
for nm, (A, B) in {'ours c256 vs vLLM': (V, R[256]), 'ours c256 vs c64': (R[256], R[64])}.items():
    k = kl21(A, B)
    print(' ', nm, ' '.join(f'[{s}: {k[lo:hi].mean():.4f}]' for s, lo, hi in segs))

print('== same document with / without the ChatML wrapper (chunk 256 vs chunk 1024, flash)')
a = load(f'{OUT}/w_flash256.safetensors'); b = load(f'{OUT}/w_flash1024.safetensors')
for nm in a:
    k = kl21(a[nm], b[nm]); ps = p_special(a[nm])
    print(f'  {nm:32s} mean {k.mean():.4f} n>1 {int((k > 1).sum()):3d}  positions with >10% turn-control mass {(ps > 0.1).mean():.1%}')
