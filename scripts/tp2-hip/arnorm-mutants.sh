#!/usr/bin/env bash
# Lane tp2 (ar-norm): build mutants of kernels/hip/ar_norm.hip as prebuilt test binaries, in the CPU slot, ON tatooine. The
# window (stages-arnorm.sh) runs them: every one must make the GPU test fail ("killed"). No GPU needed here.
#   scripts/tp2-hip/sync.sh arnorm HEAD       # puts ~/intarsiaml-tp2-arnorm there
#   fleetq run --host tatooine-r9700 --mem 0 --load cpu --expect 40 --label "tp2 arnorm mutants build" -- \
#     ssh tatooine 'bash -s' < scripts/tp2-hip/arnorm-mutants.sh
# One copy of the tree (~/intarsiaml-tp2-arnorm-mut) gets one edit at a time and is built into the arnorm hiptest volume, the
# ar_norm test binary is copied to <copy>/mutbin/<name>; the volume ends with the clean tree rebuilt (so the window's own
# hiptest run is incremental). Tatooine's disk is nearly full: this adds a source copy and nothing else.
set -uo pipefail
SRC=$HOME/intarsiaml-tp2-arnorm; MUT=$HOME/intarsiaml-tp2-arnorm-mut
VOL=intarsia-cargo-target-tp2-arnorm-hiptest
mkdir -p ~/tp2/logs "$MUT/mutbin"  # (the first loop iteration finds nothing to keep)
build() { # build <tree> <log> -> prints the test executable's path in the volume
  INTARSIA_REPO=$1 INTARSIA_TARGET_VOL=$VOL CARGO_BUILD_JOBS=${JOBS:-20} "$1/deploy/tatooine/dev.sh" \
    cargo test --release -p intarsia-backend-hip --test ar_norm --no-run > "$2" 2>&1 < /dev/null
  echo "build exit $? $(grep -aE '^error|Finished' "$2" | tail -2 | tr '\n' '|')" >&2
  grep -aoE 'target/release/deps/ar_norm-[0-9a-f]+' "$2" | tail -1
}
# name | exact text in ar_norm.hip | replacement
mutants() { cat <<'EOT'
unrounded-allreduce|ik_unpack8(y, f);|/* mutant */
unrounded-sum|ik_unpack8(p, f); /* continue from the bf16-rounded sum */|/* mutant */
other-block-size|ik_round_up((unsigned)(items < 256 ? items : 256), 32)|ik_round_up((unsigned)(items < 128 ? items : 128), 32)
woffset-ignored|f[q] = f[q] * r * (g[q] + woff);|f[q] = f[q] * r * (g[q]);
residual-dropped|for (int q = 0; q < 8; ++q) f[q] += g[q];|for (int q = 0; q < 8; ++q) f[q] += 0.f * g[q];
EOT
}
while IFS='|' read -r name old new; do
  # a fresh copy of the tree, keeping the binaries built so far
  rm -rf "$MUT.keep"; mv "$MUT/mutbin" "$MUT.keep" 2>/dev/null; rm -rf "$MUT"; cp -a "$SRC" "$MUT"; rm -rf "$MUT/tp2-out"
  mkdir -p "$MUT/mutbin"; mv "$MUT.keep"/* "$MUT/mutbin"/ 2>/dev/null; rm -rf "$MUT.keep"
  python3 - "$MUT/kernels/hip/ar_norm.hip" "$old" "$new" <<'PY' || { echo "mutant $name: pattern not found exactly once"; continue; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if s.count(old) != 1:
    sys.exit(1)
open(p, 'w').write(s.replace(old, new))
PY
  exe=$(build "$MUT" ~/tp2/logs/mut-$name.log)
  [[ -n $exe ]] || { echo "mutant $name: no test executable"; continue; }
  INTARSIA_REPO=$MUT INTARSIA_TARGET_VOL=$VOL "$MUT/deploy/tatooine/dev.sh" cp "/$exe" "/work/mutbin/$name"
  echo "mutant $name: $(ls -la "$MUT/mutbin/$name" 2>&1 | awk '{print $5}') bytes"
done < <(mutants)
# the clean tree again, so the volume matches what the window runs; and its executable, for the record
exe=$(build "$SRC" ~/tp2/logs/mut-clean.log)
echo "clean: $exe"; ls -la "$MUT/mutbin"
