#!/usr/bin/env bash
# Lane mxr, ON tatooine inside a fleetq job (--load cpu), after cbuild.sh dev: build the rows_exact test binary of the clean dev
# tree and of each deliberate kernel bug (mutant) of kernels/hip/mxfp4.hip, keep every binary under the dev target dir
# (/target/mutants/<name>/rows_exact inside dev.sh), print each binary's sha256 against the clean one. A mutation that does not
# apply exactly once fails the script. The source is restored (and the clean build re-made) on exit.
set -euo pipefail
R=$HOME/intarsiaml-mxr-dev; F=$R/kernels/hip/mxfp4.hip; T=/data/intarsiaml-mxr-target-dev
D=$R/deploy/tatooine/dev.sh
cp "$F" "$T/mxfp4.hip.clean"
restore() { cp "$T/mxfp4.hip.clean" "$F"; }
trap restore EXIT
build() { # build <name>: compile rows_exact, copy it to $T/mutants/<name>/rows_exact
  local exe
  exe=$(INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-mxr-target-dev $D cargo test --release --no-run -p intarsia-backend-hip --test rows_exact 2>&1 < /dev/null \
        | grep -aoE "/target/release/deps/rows_exact-[0-9a-f]+" | tail -1)
  [ -n "$exe" ] || { echo "BUILD FAILED for $1"; exit 1; }
  mkdir -p "$T/mutants/$1"; cp "$T/${exe#/target/}" "$T/mutants/$1/rows_exact"
  sha256sum "$T/mutants/$1/rows_exact" | cut -c1-16
}
mutate() { # mutate <old> <new>: exactly one occurrence, else fail
  python3 - "$F" "$1" "$2" <<'PY'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
n = s.count(old)
if n != 1:
    sys.exit(f"mutation does not apply: {n} occurrences of {old!r}")
open(p, "w").write(s.replace(old, new))
PY
}
clean=$(build clean); echo "clean      $clean"
# M1: the flag never reaches the kernel (flagged calls run the int8 path on out-of-range / W4A16 input)
mutate 'if (lane * 4 + j < nflags) bad |= flags[lane * 4 + j];' 'if (lane * 4 + j < nflags) bad |= 0u * flags[lane * 4 + j];'
h=$(build m1-noflag); echo "m1-noflag  $h"; [ "$h" != "$clean" ] || { echo "m1 binary equals clean"; exit 1; }; restore
# M2: the group factor one binade off (2^(sb - 128)): only the int8 path sees it
mutate 'return (sb >= 3u && sb <= 244u) ? __uint_as_float((sb - 2u) << 23) : 0.f;' 'return (sb >= 3u && sb <= 244u) ? __uint_as_float((sb - 1u) << 23) : 0.f;'
h=$(build m2-xscale); echo "m2-xscale  $h"; [ "$h" != "$clean" ] || { echo "m2 binary equals clean"; exit 1; }; restore
# M3: the activation codes' odd / even halves swapped in the pre-pass (dot4 lanes pair the wrong elements)
mutate '        xi[i] = mx_i8x8(w);' '        { const uint2 t = mx_i8x8(w); xi[i] = make_uint2(t.y, t.x); }'
h=$(build m3-pairing); echo "m3-pairing $h"; [ "$h" != "$clean" ] || { echo "m3 binary equals clean"; exit 1; }; restore
# M4: the flagged-call chain takes the elements of a chunk in reverse order (only the fp-chain path sees it)
mutate 'for (int e = 0; e < 8; ++e) t = fmaf(ik_fp4_value((ww[c] >> (4 * e)) & 15u), xf[e], t);' 'for (int e = 7; e >= 0; --e) t = fmaf(ik_fp4_value((ww[c] >> (4 * e)) & 15u), xf[e], t);'
h=$(build m4-chainorder); echo "m4-chainorder $h"; [ "$h" != "$clean" ] || { echo "m4 binary equals clean"; exit 1; }; restore
again=$(build clean2); echo "clean2     $again (clean rebuilt after the mutants; equal to clean: $([ "$again" = "$clean" ] && echo yes || echo no))"
echo "MUTANTS BUILT $(date +%T)"
