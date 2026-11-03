#!/bin/zsh
export FLEETQ_OWNER=claude-m27-gemv
cd /opt/code/intarsiaml/.worktrees/m27-gemv
S=/opt/code/intarsiaml/.worktrees/m27-gemv/.scratch; R=docs/results/2026-11-02-m27-gemv/raw; P=docs/results/raw/2026-10-15-metal-longctx-profile/prompts
fleetq run --host endor --expect 25 --mem 30 --label "m27-gemv: gates 1 on v7 vs main" -- zsh $R/scripts/gates1.sh $R $S/intarsia-v7 $S/intarsia-main $S/intarsia-main > $S/gates1d.log 2>&1
fleetq run --host endor --expect 10 --mem 12 --label "m27-gemv: kernels test v7 and main" -- zsh $R/scripts/gates2.sh $R $S/intarsia-v7 $S/intarsia-main > $S/gates2d.log 2>&1
fleetq run --host endor --expect 25 --mem 20 --load cpu --label "m27-gemv: cargo test --workspace on final head" -- env CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/m27-gemv/target-test cargo test --workspace --release --no-fail-fast > $R/kernels/cargo-test-workspace.log 2>&1
fleetq run --host endor --expect 8 --mem 8 --quiet gpu,cpu --label "m27-gemv: fused vs pre-pass microbench W4A16 and W4A4" -- zsh $S/bw6.sh $S/bw/xfuse-act.txt $S/mxfp4_bw-v7 > $S/bw/run10.log 2>&1
fleetq run --host endor --expect 8 --mem 8 --quiet gpu,cpu --label "m27-gemv: GEMV microbench main vs v7, M=1/3/8" -- zsh $S/bw2.sh $S/bw/ab-v7.txt $S/mxfp4_bw-base $S/mxfp4_bw-v7 > $S/bw/run11.log 2>&1
for L in 4096 16384; do
fleetq run --host endor --expect 18 --mem 30 --quiet gpu,cpu --label "m27-gemv: speed A/B $L, MTP off, main vs v7" -- zsh $S/abgen.sh $S/ab7-$L $P/prompt-$L.txt 128 3 main:$S/intarsia-main v7:$S/intarsia-v7 v7nofuse:$S/intarsia-v7:INTARSIA_METAL_MXFP4_XFUSE=0 v7hs3:$S/intarsia-v7:INTARSIA_METAL_ATTN_HS=3 > $S/ab7-$L.log 2>&1
done
fleetq run --host endor --expect 12 --mem 30 --quiet gpu,cpu --label "m27-gemv: speed A/B 4k MTP k2, main vs v7" -- zsh $S/abgen.sh $S/ab7-k2 $P/prompt-4096.txt 128 3 maink2:$S/intarsia-main v7k2:$S/intarsia-v7 v7hs3k2:$S/intarsia-v7:INTARSIA_METAL_ATTN_HS=3 > $S/ab7-k2.log 2>&1
fleetq run --host endor --expect 8 --mem 30 --quiet gpu,cpu --label "m27-gemv: profiles 4k decode (kernel v7, cb main + v7)" -- zsh -c "zsh $S/prof.sh $S/intarsia-v7 $S/prof/v7-4k $P/prompt-4096.txt; zsh $S/profcb.sh $S/prof/cb3 $P/prompt-4096.txt $S/intarsia-main $S/intarsia-v7" > $S/prof/run-v7.log 2>&1
echo CHAIN-DONE > $S/chain2.done
