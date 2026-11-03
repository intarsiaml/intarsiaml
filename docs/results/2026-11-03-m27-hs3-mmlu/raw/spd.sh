#!/bin/zsh
export FLEETQ_OWNER=claude-m27-hs3 SIZES=16384 GEN=128 REPS=2
W=/opt/code/intarsiaml/.worktrees/m27-hs3
fleetq run --host endor --expect 12 --mem 28 --quiet gpu,cpu --label "m27-hs3: 16k decode check, HS unset" -- zsh $W/.scratch/speed-arm.sh hs3off-16k $W/.scratch/intarsia $W/raw/speed
fleetq run --host endor --expect 12 --mem 28 --quiet gpu,cpu --label "m27-hs3: 16k decode check, HS=3" -- env INTARSIA_METAL_ATTN_HS=3 zsh $W/.scratch/speed-arm.sh hs3on-16k $W/.scratch/intarsia $W/raw/speed
