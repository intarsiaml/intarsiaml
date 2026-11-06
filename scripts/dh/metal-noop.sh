#!/usr/bin/env bash
# Lane dhm: pre-merge Metal no-op of the lane dh head (27B DFlash2 on HIP) against origin/main, ON endor inside one fleetq slot (--exclusive):
#   FLEETQ_OWNER=claude-f31ccf91-dhm fleetq run --host endor --exclusive --mem 24 --expect 90 --label "dhm metal no-op" -- scripts/dh/metal-noop.sh <base-rev>
# (1) plan show (text + json) and 256 greedy ids, Qwen3-8B and the 27B Quark MXFP4 (file tags say "35" for the second model, as in
#     scripts/tp2-hip/noop/noop.sh; here it is the 27B), short and ~3.5k prompts; plus plan show --json for the 35B NVFP4, base vs head.
# (2) kernels test (metal) on both; (3) goldens qwen3-8b-fp32 / -long (bf16 gate class) on the head; (4) cargo test --workspace on the head
#     in a test target dir separate from the metal build dirs (a --release test run in the metal dir relinks the binary without metal).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
BASE=${1:?base rev}
DEV="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
S=$DEV/.dh/dhm; O=$S/out; mkdir -p $S $O
M8=$HOME/models/Qwen3-8B; M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
rm -rf $S/base-src; mkdir -p $S/base-src && git -C $DEV archive $BASE | tar -x -C $S/base-src
for t in base dev; do
  if [ $t = base ]; then T=$S/base-src; else T=$DEV; fi
  echo "== build $t $(date +%T)"
  (cd $T && CARGO_TARGET_DIR=$S/target-$t cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  RUN="$S/target-$t/release/intarsia" bash $DEV/scripts/tp2-hip/noop/noop.sh run $O $t metal $M8 $M27 $DEV/scripts/tp2-hip/noop/prompts
  $S/target-$t/release/intarsia plan show --backend metal --model $M35 --max-seq 8192 --json < /dev/null > $O/plan-35b-$t.json 2> $O/plan-35b-$t.json.err
done
bash $DEV/scripts/tp2-hip/noop/noop.sh cmp $O base dev | tee $O/noop.txt
cmp -s $O/plan-35b-base.json $O/plan-35b-dev.json && [ -s $O/plan-35b-dev.json ] && echo "plan 35b json: byte-identical" || echo "plan 35b json: DIFFER or empty"
B=$S/target-dev/release/intarsia
echo "== kernels test metal $(date +%T)"
for t in base dev; do
  $S/target-$t/release/intarsia kernels test --backend metal < /dev/null > $O/kernels-$t.log 2>&1
  echo "$t exit $? FAIL lines: $(grep -c FAIL $O/kernels-$t.log)"; tail -2 $O/kernels-$t.log
done
echo "== goldens $(date +%T)"
for g in qwen3-8b-fp32 qwen3-8b-long-fp32; do
  $B accuracy --backend metal --model $M8 --golden /opt/code/intarsiaml/golden/$g.safetensors --mode both --gate-class bf16 < /dev/null > $O/golden-$g.log 2>&1
  echo "$g exit $?"; tail -4 $O/golden-$g.log
done
echo "== workspace tests on the head $(date +%T)"
(cd $DEV && CARGO_TARGET_DIR=$S/target-test cargo test --workspace --release --no-fail-fast > $O/cargo-test.log 2>&1; echo "cargo test exit $?"
 grep -E "^test result|FAILED|failed|panicked" $O/cargo-test.log | sort | uniq -c | sort -rn | head -20)
echo "== done $(date +%T)"
