#!/usr/bin/env bash
# Lane W4A8, ON tatooine inside a fleetq job (--load cpu), after build.sh: build the mxfp4_a8 test binary of the clean tree and of
# each deliberate kernel bug (mutant) of the W4A8 kernels, keep every binary under the lane's target volume (/target/mutants/<name>
# inside dev.sh). A mutation that does not apply exactly once fails the script. The sources are restored on exit.
#   m1 decode table: e4m3 bytes of magnitudes 1 and 1.5 swapped            (reference mismatch)
#   m2 the per-token scale dropped from the decode output                    (reference mismatch)
#   m3 row 1 of an M > 1 decode call carries a 2e-7 relative scale error     (invisible to the reference tolerance; rows == solo only)
#   m4 the per-token scale loses its lower clamp                             (all-zero row: 0/0)
#   m5 the fold GEMM's output exponent one binade off                        (reference mismatch, prefill)
set -euo pipefail
R=$HOME/intarsiaml-w4a8; K=$R/kernels/hip; V=intarsia-cargo-w4a8
D=$R/deploy/tatooine/dev.sh
mkdir -p "$R/.mutbak"
for f in ik_mx_a8.h mxfp4_a8.hip mxfp4_a8_gemm.hip; do cp "$K/$f" "$R/.mutbak/$f"; done
restore() { for f in ik_mx_a8.h mxfp4_a8.hip mxfp4_a8_gemm.hip; do cp "$R/.mutbak/$f" "$K/$f"; done; }
trap restore EXIT
build() { # build <name>: compile the mxfp4_a8 test binary, keep it as /target/mutants/<name>/mxfp4_a8, print its sha256[:16]
  INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V $D bash -c "
    exe=\$(cargo test --release --no-run -p intarsia-backend-hip --test mxfp4_a8 2>&1 < /dev/null | grep -aoE '/target/release/deps/mxfp4_a8-[0-9a-f]+' | tail -1)
    [ -n \"\$exe\" ] || { echo BUILD-FAILED; exit 3; }
    mkdir -p /target/mutants/$1 && cp \$exe /target/mutants/$1/mxfp4_a8 && sha256sum /target/mutants/$1/mxfp4_a8 | cut -c1-16" < /dev/null | tail -1
}
mutate() { # mutate <file> <old> <new>: exactly one occurrence, else fail
  python3 - "$K/$1" "$2" "$3" <<'PY'
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
run() { # run <name> <file> <old> <new>
  mutate "$2" "$3" "$4"; local h; h=$(build "$1"); echo "$1 $h"; [ "$h" != "$clean" ] || { echo "$1 binary equals clean"; exit 1; }; restore
}
run m1-table ik_mx_a8.h '0x3C383000u' '0x383C3000u'
run m2-noscale mxfp4_a8.hip 'const float t = ik_warp_sum(acc[r][m]) * xsm;' 'const float t = ik_warp_sum(acc[r][m]);'
run m3-row1 mxfp4_a8.hip 'const float t = ik_warp_sum(acc[r][m]) * xsm;' 'const float t = ik_warp_sum(acc[r][m]) * ((M > 1 && m == 1) ? xsm * 1.0000002f : xsm);'
run m4-noclamp ik_mx_a8.h 'fmaxf(__fdiv_rn(amax, kMxA8Max), kMxA8MinScale)' '__fdiv_rn(amax, kMxA8Max)'
run m5-exponent mxfp4_a8_gemm.hip '(int)b - 133' '(int)b - 132'
again=$(build clean2); echo "clean2 $again (equal to clean: $([ "$again" = "$clean" ] && echo yes || echo no))"
echo "MUTANTS BUILT $(date +%T)"
