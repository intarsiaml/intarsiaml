#!/usr/bin/env python3
"""Lane LDF: the instruction shape of one a8_gemm_kernel's main loop from a hipcc -S dump.
usage: loopshape.py <file.s> <ik_bf16|f> <suffix>   (fold kernel, 4x4 waves, 2x4 fragments, PD 2; suffix = the mangled template tail after PD,
e.g. Lb1E (SWZ true), Lb0E (SWZ false) for lane LDF's kernel, or E for main's, then the output type)
Prints one line: the loop body as runs of v_wmma / ds_load / ds_store / s_wait_* / barrier / global_load, other VALU counted."""
import re, sys
f, ot, V = sys.argv[1], sys.argv[2], sys.argv[3]
V = {"swz": "ELb1E", "pad": "ELb0E", "main": "E"}.get(V, V)
L = open(f).read().split('\n')
tag = f"ILb1ELi4ELi4ELi2ELi4ELi2{V}" + ("f" if ot == "f" else "7ik_bf16")
st = [i for i, l in enumerate(L) if re.match(r'^_Z\w*a8_gemm_kernel\w*:', l) and tag in l]
if not st: sys.exit("kernel not found: " + tag)
i0 = st[0]
j = next(i for i in range(i0 + 1, len(L)) if L[i].startswith('.Lfunc_end'))
body = L[i0:j]
w = [i for i, l in enumerate(body) if 'v_wmma' in l]
# the loop: from the label that starts the first wmma's basic block back-edge target
labels = {m.group(1): i for i, l in enumerate(body) for m in [re.match(r'^(\.LBB\d+_\d+):', l)] if m}
best = None
for i, l in enumerate(body):
    m = re.search(r's_cbranch\w*\s+(\.LBB\d+_\d+)', l)
    if m and m.group(1) in labels and labels[m.group(1)] < i and labels[m.group(1)] <= w[0] and i >= w[-1]:
        if best is None or (i - labels[m.group(1)]) < (best[1] - best[0]): best = (labels[m.group(1)], i)
if best is None: sys.exit("no loop")
seq = []
for l in body[best[0]:best[1] + 1]:
    l = l.split(';')[0].strip()
    if not l or l.startswith('.') and not l.startswith('.LBB'): continue
    t = l.split()[0]
    if t.startswith('v_wmma'): k = 'W'
    elif t.startswith('ds_load'): k = 'L' + re.sub(r'ds_load_b', '', t)
    elif t.startswith('ds_store'): k = 'S' + re.sub(r'ds_store_b', '', t)
    elif t.startswith('s_wait_dscnt') or t.startswith('s_wait_loadcnt') or t.startswith('s_wait_storecnt'): k = l.replace('s_wait_', 'w:').replace(' ', '')
    elif t.startswith('s_barrier'): k = t.replace('s_barrier_', 'BAR.')
    elif t.startswith('global_inv'): k = 'inv'
    elif t.startswith('global_load'): k = 'G' + t.replace('global_load_', '')
    elif t.startswith('v_'): k = 'v'
    elif t.startswith('s_wait_alu'): continue
    elif t.startswith('s_cbranch'): k = 'br'
    elif t.startswith('s_'): k = 's'
    else: continue
    seq.append(k)
out = []
for k in seq:
    if out and out[-1][0] == k: out[-1][1] += 1
    else: out.append([k, 1])
print(' '.join(k if n == 1 else f"{k}x{n}" for k, n in out))
