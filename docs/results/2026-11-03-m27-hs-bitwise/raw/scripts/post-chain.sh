#!/bin/zsh
# After the speed chain: exploratory split factors (HSB=6 GP 1, HSB=2 GP 4; MTP off, 4k/16k, one job each), the Metal gates, the workspace tests.
export FLEETQ_OWNER=claude-m27-hsb SIZES=4096,16384 GEN=256 REPS=3
W=/opt/code/intarsiaml/.worktrees/m27-hs-bitwise; BIN=$W/.scratch/intarsia.hsb2; D=$W/docs/results/2026-11-03-m27-hs-bitwise/raw; cd $W
while ! grep -q CHAIN-DONE $W/.scratch/speed-chain.log; do sleep 15; done
for n in 6 2; do
  fleetq run --host endor --expect 14 --mem 28 --quiet gpu,cpu --label "m27-hsb: speed off HSB=$n (exploratory) 4k/16k" -- env INTARSIA_METAL_ATTN_HSB=$n zsh $D/scripts/speed-arm.sh off-X$n $BIN $D/speed --spec off
done
fleetq run --host endor --expect 45 --mem 30 --label "m27-hsb: Metal gates (plan show, accuracy, MTP) HSB=3" -- zsh $D/scripts/gates.sh $D/gates $BIN $W/.scratch/intarsia.main-47af18777 > $W/.scratch/gates.log 2>&1
fleetq run --host endor --expect 25 --mem 20 --load cpu --label "m27-hsb: cargo test --workspace" -- env CARGO_TARGET_DIR=$W/target-test cargo test --workspace --release --no-fail-fast > $D/kernels/cargo-test-workspace.log 2>&1
echo POST-DONE
