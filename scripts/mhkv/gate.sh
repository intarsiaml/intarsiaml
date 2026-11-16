#!/bin/bash
# Lane mhkv: the MTP + prefix-store gate on Metal (tests/mtp_store_metal.rs: lane mstore's gate, plus the head-restore
# assertions and the 4k acceptance case) for one model, tests one at a time (each loads the model).
# usage (inside a fleetq slot): scripts/mhkv/gate.sh 35b|27b
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mhkv; R=$W/docs/results/2026-11-16-mhkv/raw; mkdir -p $R $W/target/mhkv-run
case $1 in
  35b) M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; X=""; GEN=160 ;;
  27b) M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; X="INTARSIA_BATCHED_MTP=1"; GEN=128 ;;
  *) echo "model?"; exit 2 ;;
esac
cd $W
( while sleep 10; do echo "$(date +%T) $(memory_pressure -Q 2>/dev/null | tail -1) $(sysctl -n vm.swapusage)"; done ) > $R/gate-$1.mem 2>&1 &
MW=$!
echo "## gate $1 $(date +%T) head $(git rev-parse --short HEAD) dirty: $(git status --short crates | wc -l | tr -d ' ')"
env INTARSIA_TEST_REAL_MODEL=1 INTARSIA_MTP_STORE=1 INTARSIA_METAL_TEST_MODEL=$M INTARSIA_TEST_GEN=$GEN \
  INTARSIA_TEST_STORE_DIR=$W/target/mhkv-run $X \
  cargo test --release --locked -p intarsia-models --test mtp_store_metal -- --nocapture --test-threads=1 > $R/gate-$1.log 2>&1
rc=$?
kill $MW
echo "## exit $rc $(date +%T)"
grep -a "verdict\|test result\|panicked\|acceptance\|MISMATCH\|control\|4k" $R/gate-$1.log | head -60
exit $rc
