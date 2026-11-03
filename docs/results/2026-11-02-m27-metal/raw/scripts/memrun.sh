#!/bin/zsh
# peak resident memory for a 16k prompt: usage memrun.sh BIN OUT
BIN=$1; O=$2; M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; P=/opt/code/intarsiaml/.worktrees/m27/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
/usr/bin/time -l $BIN generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 32 --ignore-eos --profile-every 0 --prompt "$(cat $P/prompt-16384.txt)" > $O.out 2> $O.err
grep -E "maximum resident|peak memory|finish=" $O.err
