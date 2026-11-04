#!/bin/zsh
export FLEETQ_OWNER=claude-m27-hs3 QARGS="--bench gsm8k --limit 300"
W=/opt/code/intarsiaml/.worktrees/m27-hs3; R=$W/docs/results/2026-11-03-m27-hs3-mmlu/raw/decode
fleetq run --host endor --expect 40 --mem 28 --quiet gpu,cpu --label "m27-hs3: GSM8K n=300 arm off" -- zsh $W/.scratch/qarm.sh gsm-off $W/.scratch/intarsia $R
fleetq run --host endor --expect 40 --mem 28 --quiet gpu,cpu --label "m27-hs3: GSM8K n=300 arm on HS=3" -- env INTARSIA_METAL_ATTN_HS=3 zsh $W/.scratch/qarm.sh gsm-on $W/.scratch/intarsia $R
