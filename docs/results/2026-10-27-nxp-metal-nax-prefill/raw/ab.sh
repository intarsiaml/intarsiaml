#!/bin/bash
# usage (inside fleetq slot): ab.sh OUT CTX ROUNDS "label=BIN[,ENV=V]" ...   interleaved A B A B; prints prefill ms / tok/s per run
OUT=$1; CTX=$2; R=$3; shift 3; mkdir -p $OUT
P=/opt/code/intarsiaml/.worktrees/nxp/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
for r in $(seq 1 $R); do
  for arm in "$@"; do
    label=${arm%%=*}; rest=${arm#*=}; bin=${rest%%,*}; envs=""; [ "$rest" != "$bin" ] && envs=${rest#*,}
    pre="rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)"
    ( while true; do echo "$(date +%T) $(ps -axo pid,%cpu,comm -r | sed -n 2,3p | tr -s ' ' | tr '\n' ';')"; sleep 4; done ) > $OUT/$label-r$r.load &
    lp=$!
    env ${envs//,/ } $bin generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 --ignore-eos --profile-every 0 --prompt "$(cat $P/prompt-$CTX.txt)" > $OUT/$label-r$r.out 2> $OUT/$label-r$r.err
    kill $lp 2>/dev/null
    post="rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)"
    echo "r$r $label ctx=$CTX $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $OUT/$label-r$r.err | tail -1) | pre[$pre] post[$post]"
  done
done
