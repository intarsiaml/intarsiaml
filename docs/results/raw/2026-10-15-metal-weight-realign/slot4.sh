#!/bin/zsh
# slot 4: the multirow selection test with the multi-row pair entry listed (INTARSIA_METAL_MULTIROW=1, M = 15 is above
# MR_MAX_REGISTERED), plus the wrap_host/realign tests with re-alignment switched on by default (INTARSIA_METAL_REALIGN=1).
W=/opt/code/intarsiaml-weight-realign
R=$W/docs/results/raw/2026-10-15-metal-weight-realign/slot4
mkdir -p $R
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 5 --label "metal-weight-realign: slot4 env-mode tests" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
git rev-parse HEAD
INTARSIA_METAL_MULTIROW=1 cargo test -p intarsia-backend-metal --release --test multirow > $R/multirow-env.log 2>&1; echo "multirow MULTIROW=1 exit $?"
INTARSIA_METAL_REALIGN=1 cargo test -p intarsia-backend-metal --release --test realign --test wrap_host --test fill > $R/realign-env.log 2>&1; echo "realign/wrap_host REALIGN=1 exit $?"
echo done
