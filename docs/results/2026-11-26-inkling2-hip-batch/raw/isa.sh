#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-hip-batch): the device ISA of ink_mix.hip / ink_attn.hip for main (.scratch/main-tree)
# and the branch (the lane tree), build.rs's flags (-O3 -std=c++17 -fPIC --offload-arch=gfx1201 -I kernels), device-only assembly; then per
# kernel: instruction count, VGPR / SGPR count, LDS and private segment sizes, and whether the instruction bodies are identical.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-batch"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-batch INTARSIA_CONTAINER_NAME=ink-hip-batch-build
trap 'docker rm -f ink-hip-batch-build >/dev/null 2>&1' EXIT
mkdir -p .scratch/isa logs
deploy/tatooine/dev.sh bash -c '
set -u
cd /work/.scratch/isa
for v in main branch; do
  if [ $v = main ]; then R=/work/.scratch/main-tree; else R=/work; fi
  for f in ink_mix ink_attn; do
    /opt/rocm/bin/hipcc -O3 -std=c++17 -fPIC -Wno-unused-function --offload-arch=gfx1201 -I $R/kernels --cuda-device-only -S \
      $R/kernels/hip/$f.hip -o $v-$f.s || echo "COMPILE FAIL $v $f"
  done
done
ls -la
' < /dev/null > logs/isa.log 2>&1
cat logs/isa.log | tail -12
python3 - <<'EOF'
import re, os
D = os.path.expanduser("~/intarsiaml-ink-hip-batch/.scratch/isa")
def kernels(path):
    txt = open(path).read().splitlines()
    ks, cur, body = {}, None, []
    meta = {}
    for l in txt:
        m = re.match(r'^(_Z\S+):\s*(;.*)?$', l)
        if m and cur is None:
            cur, body = m.group(1), []
            continue
        if cur is not None:
            if re.match(r'^\s*\.Lfunc_end', l) or l.startswith('.Lfunc_end'):
                ks[cur] = body; cur = None; continue
            s = l.split(';')[0].strip()
            if s and not s.startswith('.') and not s.endswith(':'):
                body.append(re.sub(r'\.LBB\d+_\d+', 'L', s))
    # descriptor fields
    for name in ks:
        m = re.search(r'\.amdhsa_kernel ' + re.escape(name) + r'\n(.*?)\.end_amdhsa_kernel', open(path).read(), re.S)
        f = {}
        if m:
            for k in ('amdhsa_next_free_vgpr', 'amdhsa_next_free_sgpr', 'amdhsa_group_segment_fixed_size', 'amdhsa_private_segment_fixed_size', 'amdhsa_kernarg_size'):
                mm = re.search(r'\.' + k + r'\s+(\d+)', m.group(1))
                f[k.replace('amdhsa_', '')] = int(mm.group(1)) if mm else None
        meta[name] = f
    return ks, meta
def key(n, branch):
    # identifier + literal template args; the branch's extra trailing RT arg (Lb0E: RT = false) is dropped, RT = true kept
    m = re.match(r'^(.*?\d+(?:ink_row_kernel|ink_attn_dec_k))I((?:L[a-z]+\d+E)+)E', n)
    if not m:
        return n
    args = re.findall(r'L[a-z]+\d+E', m.group(2))
    want = 3 if 'ink_row_kernel' in m.group(1) else 2
    if branch and len(args) == want + 1:
        if args[-1] != 'Lb0E':
            return 'RT:' + m.group(1) + ''.join(args)
        args = args[:-1]
    return m.group(1) + ''.join(args)
def rekey(ks, meta, branch):
    out, om = {}, {}
    for k in ks:
        kk = key(k, branch)
        out[kk], om[kk] = ks[k], meta[k]
    return out, om
for f in ("ink_mix", "ink_attn"):
    a, am = rekey(*kernels(f"{D}/main-{f}.s"), False)
    b, bm = rekey(*kernels(f"{D}/branch-{f}.s"), True)
    print(f"== {f}: main {len(a)} kernels, branch {len(b)} ({len(set(b) - set(a))} new)")
    same = 0
    for k in sorted(a):
        if k not in b:
            print(f"  MISSING in branch: {k}"); continue
        eq = a[k] == b[k] and am[k] == bm[k]
        same += eq
        if not eq:
            print(f"  DIFF {k[:90]}: insts {len(a[k])} -> {len(b[k])}, {am[k]} -> {bm[k]}")
    print(f"  {same} / {len(a)} of main's kernels identical in the branch (instructions and descriptor)")
EOF
