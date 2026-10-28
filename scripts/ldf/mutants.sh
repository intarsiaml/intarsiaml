#!/usr/bin/env bash
# Lane LDF, ON tatooine inside a fleetq job (--load cpu), after buildboth.sh: build the mxfp4_a8 test binary of the clean tree and of each
# deliberate bug (mutant) of the prefetch-distance-2 a8 GEMM and its dispatch, keep every binary as ~/intarsiaml-ldf/ldf-bin/mut-<name>
# (self-contained; run it in the dev container without cargo inside a GPU window: it must FAIL the a8_ tests). A mutation that does not apply
# exactly once fails the script. The source is restored on exit.
#   m1 register sets swapped     the stage stored after the WMMAs and the set refilled are exchanged       (reference mismatch)
#   m2 prologue stage            the second prologue load fetches stage 2 instead of stage 1               (reference mismatch)
#   m3 grid tail                 the M grid drops the last tile when M % tile is small                    (tail rows never written: M = 4100 etc.)
#   m4 fold shift                the fold shift of every group is one octave off                           (reference mismatch)
#   m5 row 1 scale off by 2e-7   invisible to the tolerance; only bitwise checks (hash diff, rows == solo, position) see it
#   m6 loop bound                one extra stage is consumed after the last                                (reference mismatch)
#   m7 weight chunk              the second 16 B of a weight chunk is stored over the first (swizzled layout) (reference mismatch)
#   m8 fragment chunk            the weight fragments read the neighbouring 16 B chunk                      (reference mismatch)
#   m9 activation chunk          the activation chunks are stored one chunk off                             (reference mismatch)
set -euo pipefail
R=$HOME/intarsiaml-ldf; F=$R/kernels/hip/mxfp4_a8_gemm.hip; V=intarsia-cargo-ldf; D=$R/deploy/tatooine/dev.sh
mkdir -p "$R/.mutbak" "$R/ldf-bin"; cp "$F" "$R/.mutbak/mxfp4_a8_gemm.hip"
restore() { cp "$R/.mutbak/mxfp4_a8_gemm.hip" "$F"; }
trap restore EXIT
build() { # build <name>: compile the test binary, keep it as ldf-bin/mut-<name>, print its sha256[:16]
  INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V $D bash -c "
    exe=\$(cargo test --release --no-run -p intarsia-backend-hip --test mxfp4_a8 2>&1 < /dev/null | grep -aoE '/target/release/deps/mxfp4_a8-[0-9a-f]+' | tail -1)
    [ -n \"\$exe\" ] || { echo BUILD-FAILED; exit 3; }
    cp \$exe /work/ldf-bin/mut-$1 && sha256sum /work/ldf-bin/mut-$1 | cut -c1-16" < /dev/null | tail -1
}
mutate() { python3 - "$F" "$1" "$2" <<'PY'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
n = s.count(old)
if n != 1:
    sys.exit(f"mutation does not apply: {n} occurrences of {old!r}")
open(p, "w").write(s.replace(old, new))
PY
}
clean=$(build clean); echo "clean $clean"
run() { mutate "$2" "$3"; local h; h=$(build "$1"); echo "$1 $h"; [ "$h" != "$clean" ] || { echo "$1 binary equals clean"; exit 1; }; restore; }
run m1-regsets 'step(kt0 + u, st[(u + 1) % PD], st[u])' 'step(kt0 + u, st[u], st[(u + 1) % PD])'
run m2-prologue 'gload(st[p], p * kA8BK)' 'gload(st[p], (p + 1) * kA8BK)'
run m3-gridtail '(((int64_t)a.M + WM * TM * 16 - 1) / (WM * TM * 16))' '(((int64_t)a.M + WM * TM * 16 - 17) / (WM * TM * 16))'
run m4-foldshift 'st.rt[c] = s - wr[c] + 6;' 'st.rt[c] = s - wr[c] + 5;'
run m5-row1 'ldexpf(acc[i][j][e] * xsm, (int)b - 133)' 'ldexpf(acc[i][j][e] * xsm * ((m == 1 && M > 1) ? 1.0000002f : 1.0f), (int)b - 133)'
run m6-loopbound 'if (kt0 + u < nk) step(' 'if (kt0 + u <= nk) step('
run m7-wchunk 'lo(r, ch * 2 + 1)) = make_uint4(p2.x' 'lo(r, ch * 2)) = make_uint4(p2.x'
run m8-fraglane 'lo(wn * TN * 16 + i * 16 + lr, (kk >> 4) + (lane >> 4))' 'lo(wn * TN * 16 + i * 16 + lr, ((kk >> 4) + (lane >> 4)) ^ 1)'
run m9-xchunk 'lo(idx >> 2, idx & 3)' 'lo(idx >> 2, (idx & 3) ^ 1)'
again=$(build clean2); echo "clean2 $again (equal to clean: $([ "$again" = "$clean" ] && echo yes || echo no))"
echo "MUTANTS BUILT $(date +%T)"
