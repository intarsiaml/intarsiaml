#!/usr/bin/env bash
# Lane tp2: build one CLI binary per mutant of the tp=2 graph / batching / vocab-head logic (mutate.py), in the CPU slot ON
# tatooine. Each mutant is a copy of the tree ~/intarsiaml-tp2-<SRC> (default batch) at ~/intarsiaml-tp2-mut-<name> with the
# mutated CLI as tp2-intarsia; the window (stages-mut.sh) runs them. Builds incrementally in the SRC tree's volume.
#   scp scripts/tp2-hip/mutate.py tatooine:tp2/mutate.py
#   fleetq run --host tatooine-r9700 --mem 0 --load cpu --expect 25 --label "tp2 mutants build" -- \
#     ssh tatooine 'bash -s' < scripts/tp2-hip/mutants-build.sh
set -uo pipefail
SRC=${SRC:-batch}; VOL=intarsia-cargo-target-tp2-$SRC
NAMES=${NAMES:-"no-peer-upload drop-captured-allreduce peer-ignores-flag tp-fold-outproj batch-peer-no-table vocab-no-gather"}
mkdir -p ~/tp2/logs
for n in $NAMES; do
  M=$HOME/intarsiaml-tp2-mut-$n
  rm -rf "$M"; cp -a "$HOME/intarsiaml-tp2-$SRC" "$M"; rm -rf "$M/tp2-out"
  python3 ~/tp2/mutate.py "$M" "$n" || { echo "mutant $n: not applied"; continue; }
  INTARSIA_REPO=$M INTARSIA_TARGET_VOL=$VOL "$M/deploy/tatooine/dev.sh" env CARGO_BUILD_JOBS=20 cargo build --release -p intarsia-cli --features hip \
    > ~/tp2/logs/mut-$n.log 2>&1 < /dev/null
  echo "mutant $n build exit $? $(grep -aE '^error|Finished' ~/tp2/logs/mut-$n.log | tail -2 | tr '\n' '|')"
  INTARSIA_REPO=$M INTARSIA_TARGET_VOL=$VOL "$M/deploy/tatooine/dev.sh" cp /target/release/intarsia /work/tp2-intarsia < /dev/null
done
# the volume now holds the last mutant: make the clean tree's mutated sources newer than any build output
touch "$HOME/intarsiaml-tp2-$SRC/crates/intarsia-core/src/model.rs" "$HOME/intarsiaml-tp2-$SRC/crates/intarsia-core/src/model/graph_step.rs"
ls -la ~/intarsiaml-tp2-mut-*/tp2-intarsia; df -h / | tail -1
