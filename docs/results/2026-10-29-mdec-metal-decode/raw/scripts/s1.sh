#!/bin/bash
# MDEC short check: 35B 4k decode, variants via env. usage: s1.sh BIN OUT [ctx] -- tag:ENV=V,ENV=V ...
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
BIN=$1; O=$2; CTX=$3; shift 3; mkdir -p $O
W=/opt/code/intarsiaml/.worktrees/mdec; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
case $CTX in 1k) PF=$W/docs/results/raw/2026-10-22-md-metal-35b-decode-profile/prompt-1k.txt;; *) PF=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-$CTX.txt;; esac
for a in "$@"; do
  tag=${a%%:*}; envs=${a#*:}; [ "$envs" = "$a" ] && envs=""
  env X=1 ${envs//,/ } $BIN generate --model $M35 --backend metal --max-seq 20000 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos \
    --prompt "$(cat $PF)" > $O/$tag.out 2> $O/$tag.err
  echo "$tag [$envs] $(grep -o 'decode [0-9]* ms ([0-9.]* tok/s)' $O/$tag.err | tail -1) md5 $(md5 -q $O/$tag.out)"
done
