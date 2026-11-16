#!/bin/zsh
# Lane mnr no-op gates (one fleetq slot): scripts/dm/noop.sh (plan show --json for 8B / 27B / 35B x default, --spec off, --spec mtp
# and kernels list / test, lane binary against a build of origin/main, knob unset), then `kernels test` and `kernels list` of the
# lane binary with INTARSIA_METAL_NAX_ROWS=1 (the new rows must be ADMITTED).
#   fleetq run --host endor --expect 25 --mem 30 --quiet gpu --label "mnr: no-op gates" -- scripts/mnr/noop.sh <lane bin> <base bin>
set -u
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
cd ${0:A:h}/../..
W=$PWD
# build the lane binary from the tree inside this slot when it is stale (CPU slots queue behind the timing runs tonight)
(cd $W && cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -1 && mkdir -p target-bin && cp target/release/intarsia target-bin/intarsia-mnr)
B=${1:?lane bin}; A=${2:?base bin}; O=docs/results/2026-11-15-mnr-metal-nax-rows/raw/noop; mkdir -p $O
zsh scripts/dm/noop.sh $B $A $O
INTARSIA_METAL_NAX_ROWS=1 $B kernels list --backend metal > $O/klist-knob.txt 2>&1 < /dev/null
INTARSIA_METAL_NAX_ROWS=1 $B kernels test --backend metal > $O/ktest-knob.txt 2>&1 < /dev/null
echo "kernels test knob exit $? $(tail -1 $O/ktest-knob.txt | cut -c1-160)" | tee -a $O/steps.log
diff $O/klist-lane.txt $O/klist-knob.txt | grep '^[<>]' | tee -a $O/steps.log
grep -n "naxr" $O/ktest-knob.txt | cut -c1-200 | tee -a $O/steps.log
