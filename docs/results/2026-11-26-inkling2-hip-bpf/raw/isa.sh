#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-hip-bpf): the device ISA of the kernel files this lane touches, main
# (.scratch/main-tree = git archive 80719483a kernels) vs the branch, build.rs's flags; then per kernel: instructions and descriptor
# (VGPR / SGPR / LDS / private / kernarg) identical or not. The branch's ink_row_kernel has one more template argument (SEG): its
# trailing `Lb0E` (SEG = false) is dropped before matching; SEG = true instantiations count as new.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-bpf"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-bpf INTARSIA_CONTAINER_NAME=ink-hip-bpf-build
trap 'docker rm -f ink-hip-bpf-build >/dev/null 2>&1' EXIT
mkdir -p .scratch/isa logs
[ -n "${SKIP_COMPILE:-}" ] || deploy/tatooine/dev.sh bash -c '
set -u
cd /work/.scratch/isa
for v in main branch; do
  if [ $v = main ]; then R=/work/.scratch/main-tree; else R=/work; fi
  for f in ink_mix ink_mix_rows ink_moe ink_mix_segs; do
    [ -f $R/kernels/hip/$f.hip ] || continue
    /opt/rocm/bin/hipcc -O3 -std=c++17 -fPIC -Wno-unused-function --offload-arch=gfx1201 -I $R/kernels --cuda-device-only -S \
      $R/kernels/hip/$f.hip -o $v-$f.s || echo "COMPILE FAIL $v $f"
  done
done
ls -la
' < /dev/null > logs/isa.log 2>&1
tail -12 logs/isa.log
python3 - <<'PY'
import re, os
D = os.path.expanduser("~/intarsiaml-ink-hip-bpf/.scratch/isa")
def kernels(path):
    src = open(path).read()
    ks, meta, cur, body = {}, {}, None, []
    for l in src.splitlines():
        m = re.match(r'^(_Z\S+):\s*(;.*)?$', l)
        if m and cur is None:
            cur, body = m.group(1), []
            continue
        if cur is not None:
            if l.lstrip().startswith('.Lfunc_end'):
                ks[cur] = body; cur = None; continue
            s = l.split(';')[0].strip()
            if s and not s.startswith('.') and not s.endswith(':'):
                body.append(re.sub(r'\.LBB\d+_\d+', 'L', s))
    for name in ks:
        m = re.search(r'\.amdhsa_kernel ' + re.escape(name) + r'\n(.*?)\.end_amdhsa_kernel', src, re.S)
        f = {}
        if m:
            for k in ('next_free_vgpr', 'next_free_sgpr', 'group_segment_fixed_size', 'private_segment_fixed_size', 'kernarg_size'):
                mm = re.search(r'\.amdhsa_' + k + r'\s+(\d+)', m.group(1))
                f[k] = int(mm.group(1)) if mm else None
        meta[name] = f
    return ks, meta
def key(n):
    # identifier + literal template arguments only (the parameter type's mangling names the template's own conditional, which
    # nests one more level in the branch); a fifth argument is SEG: Lb0E (false) dropped, Lb1E kept apart
    m = re.match(r'^(.*?\d+ink_row_kernel)I((?:L[a-z]+\d+E)+)E', n)
    if not m:
        return n
    args = re.findall(r'L[a-z]+\d+E', m.group(2))
    if len(args) == 5:
        if args[-1] != 'Lb0E':
            return 'SEG:' + m.group(1) + ''.join(args)
        args = args[:-1]
    return m.group(1) + ''.join(args)
for f in ("ink_mix", "ink_mix_rows", "ink_moe"):
    pa, pb = f"{D}/main-{f}.s", f"{D}/branch-{f}.s"
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"== {f}: missing"); continue
    a0, am0 = kernels(pa)
    a = {key(k): v for k, v in a0.items()}
    am = {key(k): v for k, v in am0.items()}
    b0, bm0 = kernels(pb)
    b = {key(k): v for k, v in b0.items()}
    bm = {key(k): v for k, v in bm0.items()}
    new = sorted(set(b) - set(a))
    print(f"== {f}: main {len(a)} kernels, branch {len(b)} ({len(new)} new: {[n[:70] for n in new][:4]})")
    same = 0
    for k in sorted(a):
        if k not in b:
            print(f"  MISSING in branch: {k}"); continue
        eq = a[k] == b[k] and am[k] == bm[k]
        same += eq
        if not eq:
            print(f"  DIFF {k[:90]}: insts {len(a[k])} -> {len(b[k])}, {am[k]} -> {bm[k]}")
    print(f"  {same} / {len(a)} of main's kernels identical in the branch (instructions and descriptor)")
if os.path.exists(f"{D}/branch-ink_mix_segs.s"):
    s, _ = kernels(f"{D}/branch-ink_mix_segs.s")
    print(f"== ink_mix_segs (new TU): {len(s)} kernels, {sum(1 for k in s if key(k).startswith('SEG:'))} SEG instantiations")
PY
