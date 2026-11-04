#!/bin/zsh
# Post-merge build of this lane (intarsia.rtm), then the gates (GPU, not quiet) and the workspace tests. Owner claude-m27-hsr.
export FLEETQ_OWNER=claude-m27-hsr
W=/opt/code/intarsiaml/.worktrees/m27-hs-rowtable; R=$W/docs/results/2026-11-03-m27-hs-rowtable/raw; cd $W
fleetq run --host endor --load cpu --expect 20 --mem 20 --label "m27 rowtable: post-merge build intarsia.rtm (cpu)" -- zsh $R/scripts/build.sh intarsia.rtm > $R/build-rtm.log 2>&1
fleetq run --host endor --load cpu --expect 30 --mem 20 --label "m27 rowtable: cargo test --workspace --release (cpu)" -- env CARGO_TARGET_DIR=$W/target-test cargo test --workspace --release --no-fail-fast > $R/cargo-test-workspace.log 2>&1
fleetq run --host endor --expect 90 --mem 30 --label "m27 rowtable: gates (ids, kernels, plan show, accuracy, MTP)" -- zsh $R/scripts/gates-rt.sh $R/gates $W/.scratch/intarsia.rtm $W/.scratch/intarsia.def > $R/gates-summary.log 2>&1
echo CHAIN-GATES-DONE >> $R/gates-summary.log
