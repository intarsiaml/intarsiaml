#!/usr/bin/env bash
# Lane pf: Metal no-op and workspace tests on endor, ON endor inside one fleetq slot (--exclusive):
#   fleetq run --host endor --exclusive --mem 40 --expect 110 --label "pf: Metal no-op + cargo test --workspace" -- scripts/cpf/metal-noop.sh
# (1) no-op vs origin/main 5390e65b4: plan show (text + json) and 256 greedy ids, Qwen3-8B and the 35B NVFP4, short and ~3.5k prompts
#     (scripts/tp2-hip/noop/noop.sh); kernels test; goldens qwen3-8b-fp32 / -long (bf16 gate class) on the branch.
# (2) cargo test --workspace --no-fail-fast on the branch.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
DEV="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
BASE=${BASE:-5390e65b4}
S=$DEV/.scratch/metal; O=$DEV/docs/results/2026-11-05-cuda-pack-flip/raw/metal; mkdir -p $S $O
M8=$HOME/models/Qwen3-8B; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
if [ ! -d $S/base-src ]; then mkdir -p $S/base-src && git -C $DEV archive $BASE | tar -x -C $S/base-src; fi
for t in base dev; do
  if [ $t = base ]; then T=$S/base-src; else T=$DEV; fi
  echo "== build $t $(date +%T)"
  (cd $T && CARGO_TARGET_DIR=$S/target-$t cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  RUN="$S/target-$t/release/intarsia" bash $DEV/scripts/tp2-hip/noop/noop.sh run $O $t metal $M8 $M35 $DEV/scripts/tp2-hip/noop/prompts
done
bash $DEV/scripts/tp2-hip/noop/noop.sh cmp $O base dev | tee $O/noop.txt
B=$S/target-dev/release/intarsia
echo "== kernels test metal $(date +%T)"
$B kernels test --backend metal < /dev/null > $O/kernels.log 2>&1; echo "kernels exit $?"; tail -3 $O/kernels.log
echo "== goldens $(date +%T)"
for g in qwen3-8b-fp32 qwen3-8b-long-fp32; do
  $B accuracy --backend metal --model $M8 --golden /opt/code/intarsiaml/golden/$g.safetensors --mode both --gate-class bf16 < /dev/null > $O/golden-$g.log 2>&1
  echo "$g exit $?"; tail -4 $O/golden-$g.log
done
echo "== cargo test --workspace $(date +%T)"
(cd $DEV && CARGO_TARGET_DIR=$S/target-test cargo test --workspace --no-fail-fast < /dev/null > $O/cargo-test.log 2>&1; echo "cargo test exit $?" | tee $O/cargo-test-exit.txt)
grep -aE '^test result' $O/cargo-test.log | awk '{p+=$4; f+=$6; i+=$8} END {print "passed", p, "failed", f, "ignored", i, "binaries", NR}' | tee $O/cargo-test-sum.txt
grep -aE '^test .* FAILED|^    [a-z_:]+$' $O/cargo-test.log | head -20
echo "== done $(date +%T)"
