#!/bin/zsh
# Lane mnr final checks on the finished tree (one fleetq slot): tests/mxfp4_naxr.rs with the knob unset and set, then
# cargo test --workspace --release --locked.
#   fleetq run --host endor --expect 25 --mem 12 --quiet gpu --label "mnr: final tests" -- scripts/mnr/final.sh
set -u
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
cd ${0:A:h}/../..
R=docs/results/2026-11-15-mnr-metal-nax-rows/raw
for knob in unset 1; do
  E=(); [[ $knob == 1 ]] && E=(INTARSIA_METAL_NAX_ROWS=1)
  env X=1 $E cargo test --release --locked -p intarsia-backend-metal --test mxfp4_naxr -- --nocapture > $R/test-knob-$knob.log 2>&1
  echo "tests knob $knob: exit $? $(grep -E '^test result' $R/test-knob-$knob.log)"
done
cargo test --workspace --release --locked > $R/workspace-test.log 2>&1
echo "workspace exit $? ; $(grep -c '^test result: ok' $R/workspace-test.log) ok suites, $(grep -c '^test result: FAILED' $R/workspace-test.log) failed suites"
grep -E "^test .* FAILED|panicked" $R/workspace-test.log | head -10
