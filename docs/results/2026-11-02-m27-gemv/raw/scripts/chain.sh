#!/bin/zsh
export FLEETQ_OWNER=claude-m27-gemv
cd /opt/code/intarsiaml/.worktrees/m27-gemv
S=/opt/code/intarsiaml/.worktrees/m27-gemv/.scratch; R=docs/results/2026-11-02-m27-gemv/raw; P=docs/results/raw/2026-10-15-metal-longctx-profile/prompts
fleetq run --host endor --expect 25 --mem 30 --label "m27-gemv: gates 1 on v5 vs main" -- zsh $R/scripts/gates1.sh $R $S/intarsia-v5 $S/intarsia-main $S/intarsia-main > $S/gates1c.log 2>&1
fleetq run --host endor --expect 10 --mem 12 --label "m27-gemv: kernels test v5 and main" -- zsh $R/scripts/gates2.sh $R $S/intarsia-v5 $S/intarsia-main > $S/gates2c.log 2>&1
fleetq run --host endor --expect 25 --mem 20 --load cpu --label "m27-gemv: cargo test --workspace on merged head" -- env CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/m27-gemv/target-test cargo test --workspace --release --no-fail-fast > $R/kernels/cargo-test-workspace.log 2>&1
for L in 4096 16384; do
fleetq run --host endor --expect 14 --mem 30 --quiet gpu,cpu --label "m27-gemv: speed A/B $L, MTP off, main vs v5" -- zsh $S/abgen.sh $S/ab5-$L $P/prompt-$L.txt 128 3 main:$S/intarsia-main v5:$S/intarsia-v5 v5hs3:$S/intarsia-v5:INTARSIA_METAL_ATTN_HS=3 > $S/ab5-$L.log 2>&1
done
fleetq run --host endor --expect 10 --mem 30 --quiet gpu,cpu --label "m27-gemv: speed A/B 4k MTP k2, main vs v5" -- zsh $S/abgen.sh $S/ab5-k2 $P/prompt-4096.txt 128 3 maink2:$S/intarsia-main v5k2:$S/intarsia-v5 v5hs3k2:$S/intarsia-v5:INTARSIA_METAL_ATTN_HS=3 > $S/ab5-k2.log 2>&1
fleetq run --host endor --expect 8 --mem 8 --quiet gpu,cpu --label "m27-gemv: GEMV microbench main vs v5, M=1/3/8" -- zsh $S/bw2.sh $S/bw/ab-v5.txt $S/mxfp4_bw-base $S/mxfp4_bw-v5 > $S/bw/run8.log 2>&1
fleetq run --host endor --expect 8 --mem 30 --quiet gpu,cpu --label "m27-gemv: profiles 4k decode (kernel v5, cb main + v5)" -- zsh -c "zsh $S/prof.sh $S/intarsia-v5 $S/prof/v5-4k $P/prompt-4096.txt; zsh $S/profcb.sh $S/prof/cb2 $P/prompt-4096.txt $S/intarsia-main $S/intarsia-v5" > $S/prof/run-v5.log 2>&1
echo CHAIN-DONE > $S/chain.done
