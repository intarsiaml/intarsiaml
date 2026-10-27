#!/usr/bin/env bash
# Lane htk: CPU-slot build ON tatooine of the htk-* mutants of kernels/hip/topk.hip (scripts/hip-decode/mutate.py), one test executable each
# (~/intarsiaml-htk/src/.bin/topk-<mutant>). Copies the tree to ~/intarsiaml-htk/mut, applies one mutant at a time, builds tests/topk.rs.
# mutrun.sh (a GPU window) then runs them: every one must FAIL.
#   fleetq run --host tatooine --load cpu --expect 40 --label "htk mutants build" -- ssh tatooine 'bash -s' < scripts/htk/mutbuild.sh
set -uo pipefail
R=$HOME/intarsiaml-htk
SRC=$R/src
MUT=$R/mut
export INTARSIA_REPO=$MUT INTARSIA_TARGET_VOL=intarsia-cargo-htk
D=$SRC/deploy/tatooine/dev.sh
mkdir -p $R/logs $SRC/.bin $MUT/.bin
for m in ${MUTANTS:-htk-tiebreak htk-slice-end htk-stage2-n htk-nanflag htk-nonstrict htk-exhausted htk-nomemset htk-kv htk-scratch}; do
  echo "== $m $(date +%T)"
  rsync -a --delete --exclude .bin $SRC/ $MUT/
  python3 $SRC/scripts/hip-decode/mutate.py $MUT $m || continue
  find $MUT -path $MUT/.bin -prune -o -type f -exec touch {} +
  $D env CARGO_BUILD_JOBS=${JOBS:-20} cargo test --release -p intarsia-backend-hip --test topk --no-run > $R/logs/mut-$m.log 2>&1 < /dev/null
  echo "build $m exit $? $(date +%T)"
  exe=$(grep -E "Executable tests/topk.rs" $R/logs/mut-$m.log | sed -E 's/.*\((.*)\)/\1/' | head -1); exe=${exe#/target/}; exe=${exe#target/}
  [ -n "$exe" ] && $D cp /target/$exe /work/.bin/topk-$m < /dev/null
  # the MUT tree is a mount of INTARSIA_REPO=$MUT: its .bin is separate, so copy back
  [ -e $MUT/.bin/topk-$m ] && cp $MUT/.bin/topk-$m $SRC/.bin/topk-$m
done
ls -la $SRC/.bin
