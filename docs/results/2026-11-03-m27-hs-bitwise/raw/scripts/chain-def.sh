#!/bin/zsh
# Shipped-default gates chain on endor (default binary .scratch/intarsia.def, HSB_DEFAULT = 3, env unset), owner claude-m27-hsb.
export FLEETQ_OWNER=claude-m27-hsb
W=/opt/code/intarsiaml/.worktrees/m27-hs-bitwise; D=$W/docs/results/2026-11-03-m27-hs-bitwise/raw; DEF=$W/.scratch/intarsia.def; BASE=$W/.scratch/intarsia.main-47af18777; cd $W
fleetq run --host endor --expect 75 --mem 30 --label "m27-hsb: default-on gates (ids, kernels, plan show, accuracy, MTP)" -- zsh $D/scripts/gates-def.sh $D/gates-def $DEF $BASE > $D/gates-def-summary.log 2>&1
for a in base1:$BASE:X=1 base2:$BASE:X=1 def:$DEF:X=1 optout:$DEF:INTARSIA_METAL_ATTN_HSB=0; do
  t=${a%%:*}; r=${a#*:}; b=${r%%:*}; e=${r#*:}
  fleetq run --host endor --expect 15 --mem 30 --label "m27-hsb: batched serve --max-batch 2, $t" -- env $e zsh $D/scripts/batched.sh $D/batched $t $b >> $D/batched-summary.log 2>&1
done
export SIZES=16384 GEN=256 REPS=3
for a in A1:$BASE B1:$DEF A2:$BASE B2:$DEF; do
  t=${a%%:*}; b=${a#*:}
  fleetq run --host endor --expect 12 --mem 28 --quiet gpu,cpu --label "m27-hsb: final speed 16k MTP k2 default, $t (${b:t})" -- env X=1 zsh $D/scripts/speed-arm.sh def-k2-$t $b $D/speed-def
done
fleetq run --host endor --expect 25 --mem 20 --load cpu --label "m27-hsb: cargo test --workspace (default-on)" -- env CARGO_TARGET_DIR=$W/target-test cargo test --workspace --release --no-fail-fast > $D/kernels/cargo-test-workspace-def.log 2>&1
echo CHAIN-DEF-DONE
