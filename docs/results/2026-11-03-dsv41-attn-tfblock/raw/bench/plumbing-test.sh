#!/bin/bash
# Offline test of attn-tfblock-bench.sh's PLUMBING (not of any kernel): the real driver and remote halves run end to end against a fake spark
# (fake ssh into a temp home, real tar pipes), fake nvcc / cuobjdump / nvidia-smi, and fake bench binaries that print canned output taken from the
# RECORDED runs (ab-run-1.log, bench-vec0.txt; the "variants" simply replay the recorded 2blk variant). It proves: the job lists build in the
# right order with the right flags, the loops/logs/tables/verdicts work, calibration failure stops with exit 3, the logs are fetched and the
# remote tree is removed on success and on failure, and the script refuses to run outside a fleetq slot. The numbers it prints are replays, not
# measurements. usage: plumbing-test.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../../../.." && pwd)
F=$ROOT/docs/results/2026-11-03-dsv41-attn-fusion/raw/bench
W=$ROOT/.worktrees/plumbing; rm -rf "$W"; mkdir -p "$W"/{bin,fbin,canned,fakehome,drv/docs/results/2026-11-03-dsv41-attn-tfblock/raw}; trap 'rm -rf "$W"' EXIT
cp -R "$HERE" "$W/drv/docs/results/2026-11-03-dsv41-attn-tfblock/raw/bench"; ln -s "$ROOT/kernels" "$W/drv/kernels"
python3 -I - "$F" "$W/canned" <<'EOF'
import re, sys
F, E = sys.argv[1] + '/', sys.argv[2] + '/'
blocks, cur = {}, None
for l in open(F + 'ab-run-1.log').read().split('\n'):
    m = re.match(r'## pair (\d+) (\w+)', l)
    if m: cur = (m.group(1), m.group(2)); blocks[cur] = []; continue
    if cur: blocks[cur].append(l)
base = '\n'.join(blocks[('1', 'base')]); var = '\n'.join(blocks[('1', 'var')])
open(E + 'ab_base', 'w').write(base + '\n'); open(E + 'ab_e2', 'w').write(var + '\n'); open(E + 'ab_e2e5', 'w').write(var + '\n')
open(E + 'ab_e5', 'w').write(var.replace('active blocks per SM 2', 'active blocks per SM 1') + '\n')
rope = ['rope_ip_bench mode 0 (out of place) iters 20 reps 9']
for l in open(F + 'bench-vec0.txt'):
    m = re.match(r'rope (.+?)\s+hash (\w+)\s+([\d.]+) ms', l)
    if m: rope.append(f"rope {m.group(1):<28s} hash {m.group(2)} min {m.group(3)} med {m.group(3)} ms")
for n in ('rope_base', 'rope_ip', 'rope_ipshfl'): open(E + n, 'w').write('\n'.join(rope) + '\n')
EOF
cat > "$W/bin/nvcc" <<'EOF'
#!/bin/bash
out=""; prev=""; for a in "$@"; do [ "$prev" = -o ] && out=$a; prev=$a; done
echo "nvcc $*" >> "$FAKE_LOG"
if [ -n "${FAKE_FAIL:-}" ] && [[ "$*" == *"$FAKE_FAIL"* ]]; then echo "fake nvcc: error: simulated compile failure" ; exit 1; fi
[ -z "$out" ] && { echo "Cuda compilation tools (FAKE nvcc)"; exit 0; }
case "$(basename "$out")" in
  ab_*|rope_*) printf '#!/bin/bash\ncat %s/%s\n' "$FAKE_CANNED" "$(basename "$out")" > "$out"; chmod +x "$out";;
  *) echo "fake object of $*" > "$out";;
esac
EOF
cat > "$W/bin/cuobjdump" <<'EOF'
#!/bin/bash
if [ "$1" = -res-usage ]; then echo "  Function attn_sparse_mma_fake:"; echo "    REG:107 STACK:0 SHARED:1792 LOCAL:0 CONSTANT[0]:1024"; else printf '\tFunction : _Z16rope_tail_kernelFAKE\n /*0010*/ FFMA R0, R1, R2, R3 ;\n /*0020*/ FMUL R4, R5, R6 ;\n'; fi
EOF
printf '#!/bin/bash\necho "GPU 0: FAKE"\n' > "$W/bin/nvidia-smi"
cat > "$W/fbin/ssh" <<'EOF'
#!/bin/bash
while [ "${1:0:1}" = - ]; do shift; done; shift
cd "$FAKE_HOME" && HOME="$FAKE_HOME" exec bash -c "$*"
EOF
chmod +x "$W"/bin/* "$W"/fbin/*; printf 'MemAvailable:   90000000 kB\n' > "$W/meminfo"
export FAKE_HOME=$W/fakehome FAKE_LOG=$W/fake-nvcc.log FAKE_CANNED=$W/canned MEMINFO=$W/meminfo
D=$W/drv/docs/results/2026-11-03-dsv41-attn-tfblock/raw/bench/attn-tfblock-bench.sh
echo "### 1. without a fleetq slot (must refuse, exit 2)"; set +e
env PATH="$W/fbin:$W/bin:$PATH" bash "$D" 2>&1 | head -3 | cut -c1-200; echo "exit ${PIPESTATUS[0]}"
echo "### 2. normal run (driver + remote halves, fake spark)"
env PATH="$W/fbin:$W/bin:$PATH" FLEETQ_JOB=999 bash "$D" > "$W/run1.txt" 2>&1; echo "exit $?"
grep -E "^\[|VERDICT|CALIBRATION|^removed|E5 on top" "$W/run1.txt" | cut -c1-150
echo "remote home after the run: [$(ls "$FAKE_HOME")]  (empty = ~/intarsia-dsv41-tfblock removed)"
echo "logs fetched: $(ls "$W/drv/docs/results/2026-11-03-dsv41-attn-tfblock/raw/run" | wc -l) files"
echo "nvcc invocations: $(grep -c . "$FAKE_LOG"); distinct -D flag sets:"; grep -o -E -- '-D[A-Z_0-9]+' "$FAKE_LOG" | sort | uniq -c
reset() { rm -rf "$W/canned"; cp -R "$W/canned.orig" "$W/canned"; rm -rf "$W/drv/docs/results/2026-11-03-dsv41-attn-tfblock/raw/run"; }
sub() { python3 -I - "$1" "$2" "$3" <<'EOF'
import sys
p, a, b = sys.argv[1:4]; s = open(p).read(); assert a in s; open(p, 'w').write(s.replace(a, b))
EOF
}
RUN="$W/drv/docs/results/2026-11-03-dsv41-attn-tfblock/raw/run"
cp -R "$W/canned" "$W/canned.orig"
echo "### 3. attention calibration failure (one recorded attention hash altered; must stop with exit 3, still fetch logs and remove the remote tree)"
reset; sub "$W/canned/ab_base" aa254a578fedfa25 aa254a578fedfa26
env PATH="$W/fbin:$W/bin:$PATH" FLEETQ_JOB=999 bash "$D" > "$W/run2.txt" 2>&1; echo "exit $?"
grep -E "FAIL|CALIBRATION" "$W/run2.txt" | cut -c1-150
echo "remote home after the failed run: [$(ls "$FAKE_HOME")]; fetched: $(ls "$RUN" | wc -l) files, incl. calib.txt: $(ls "$RUN" | grep -c '^calib.txt$')"
echo "### 4. rope hashes differ from the recorded ones but variants equal the in-job base (must NOT stop the attention experiments; E3 gets a note, not NEGATIVE)"
reset; for n in rope_base rope_ip rope_ipshfl; do sub "$W/canned/$n" d5ab75e0c17ef387 0000000000000001; done
env PATH="$W/fbin:$W/bin:$PATH" FLEETQ_JOB=999 bash "$D" > "$W/run3.txt" 2>&1; echo "exit $?"
grep -E "note rope|CALIBRATION|VERDICT" "$W/run3.txt" | cut -c1-230
echo "### 5. one rope variant differs from the in-job base (that variant NEGATIVE, the other fine, attention untouched)"
reset; sub "$W/canned/rope_ip" a163e645ec6f9b6d a163e645ec6f9b6e
env PATH="$W/fbin:$W/bin:$PATH" FLEETQ_JOB=999 bash "$D" > "$W/run4.txt" 2>&1; echo "exit $?"
grep -E "VERDICT (ip|ipshfl)" "$W/run4.txt" | cut -c1-200
echo "### 6. one variant fails to build (e2e5; must be reported and dropped, the others still run; exit 6)"
reset; FAKE_FAIL="-DATTN_TFBLOCK -DATTN_SM4" env PATH="$W/fbin:$W/bin:$PATH" FLEETQ_JOB=999 FAKE_FAIL="-DATTN_TFBLOCK -DATTN_SM4" bash "$D" > "$W/run5.txt" 2>&1; echo "exit $?"
grep -E "BUILD FAILED|^\[.*built:|VERDICT|BUILD FAILURES" "$W/run5.txt" | cut -c1-200
echo "remote home after the last run: [$(ls "$FAKE_HOME")]"
