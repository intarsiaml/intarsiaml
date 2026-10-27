#!/usr/bin/env bash
# Lane htk: CPU-slot build ON tatooine of ~/intarsiaml-htk/src (scripts/htk/sync.sh) in the HIP dev image, cargo volume intarsia-cargo-htk.
# Builds the topk / admission test binaries and the release CLI (kept as ~/intarsiaml-htk/src/.bin/intarsia). No GPU.
#   fleetq run --host tatooine --load cpu --expect 40 --label "htk build" -- ssh tatooine 'bash -s' < scripts/htk/build.sh
set -uo pipefail
R=$HOME/intarsiaml-htk
export INTARSIA_REPO=$R/src INTARSIA_TARGET_VOL=intarsia-cargo-htk
D=$INTARSIA_REPO/deploy/tatooine/dev.sh
mkdir -p $R/logs $INTARSIA_REPO/.bin
find $INTARSIA_REPO -path $INTARSIA_REPO/.bin -prune -o -type f -exec touch {} +
echo "== tests $(date +%T)"
$D env CARGO_BUILD_JOBS=${JOBS:-20} cargo test --release -p intarsia-backend-hip --test topk --test admission --test quick_coverage -p intarsia-models --test batch_cuda_real --no-run > $R/logs/build-tests.log 2>&1 < /dev/null
echo "tests exit $? $(date +%T)"; grep -E '^error|Finished|warning: unused' -A6 $R/logs/build-tests.log | tail -40
# keep the test executables (cargo names them with a hash): .bin/<test>
for t in topk admission quick_coverage batch_cuda_real; do
  exe=$(grep -E "Executable tests/$t.rs" $R/logs/build-tests.log | sed -E 's/.*\((.*)\)/\1/' | head -1)
  [ -n "$exe" ] && exe=${exe#/target/}; exe=${exe#target/}
  [ -n "$exe" ] && $D cp /target/$exe /work/.bin/$t < /dev/null
done
echo "== quick_coverage (no GPU) $(date +%T)"
$D cargo test --release -p intarsia-backend-hip --test quick_coverage > $R/logs/quick.log 2>&1 < /dev/null; echo "quick_coverage exit $?"; tail -5 $R/logs/quick.log
echo "== cli $(date +%T)"
$D env CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > $R/logs/build-cli.log 2>&1 < /dev/null
echo "cli exit $? $(date +%T)"; grep -E '^error|Finished' $R/logs/build-cli.log | tail -3
$D cp /target/release/intarsia /work/.bin/intarsia < /dev/null
ls -la $INTARSIA_REPO/.bin
