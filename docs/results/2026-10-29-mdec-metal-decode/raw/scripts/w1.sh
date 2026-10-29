#!/bin/bash
# MDEC window 1: attribution. Run inside a fleetq slot. usage: w1.sh BIN OUT
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
BIN=$1; O=$2; mkdir -p $O
W=/opt/code/intarsiaml/.worktrees/mdec
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M8=$HOME/models/Qwen3-8B
P=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
cp $W/docs/results/raw/2026-10-22-md-metal-35b-decode-profile/prompt-1k.txt $O/prompt-1k.txt
pf() { case $1 in 1k) echo $O/prompt-1k.txt;; 4k) echo $P/prompt-4096.txt;; 16k) echo $P/prompt-16384.txt;; esac; }
gen() { # tag model ctx extra-args... -- env...
  local tag=$1 m=$2 ctx=$3; shift 3; local args=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do args+=("$1"); shift; done; [ $# -gt 0 ] && shift
  echo "## $tag $(date +%T) rustc=$(pgrep -x rustc | wc -l)"
  env X=1 "$@" $BIN generate --model $m --backend metal --max-seq 20000 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos ${args[@]+"${args[@]}"} \
    --prompt "$(cat $(pf $ctx))" > $O/$tag.out 2> $O/$tag.err
  grep -iE 'decode|prefill|accept|spec' $O/$tag.err | tail -4
}
for c in 1k 4k 16k; do
  gen 35-$c-plain $M35 $c
  gen 35-$c-cb $M35 $c -- INTARSIA_METAL_PROFILE=cb:$O/35-$c-cb.jsonl
  gen 35-$c-ce1 $M35 $c -- INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/35-$c-ce1.jsonl
done
gen 35-4k-plain2 $M35 4k
gen 35-4k-mtp $M35 4k --spec mtp
gen 35-4k-mtpcb $M35 4k --spec mtp -- INTARSIA_METAL_PROFILE=cb:$O/35-4k-mtpcb.jsonl
gen 35-1k-mtp $M35 1k --spec mtp
gen 8-4k-plain $M8 4k
gen 8-4k-cb $M8 4k -- INTARSIA_METAL_PROFILE=cb:$O/8-4k-cb.jsonl
gen 8-4k-ce1 $M8 4k -- INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/8-4k-ce1.jsonl
# served batched: plain baseline, then host profile + cb profile
PORT=18471
srv() { # tag conc gen envs...
  local tag=$1 conc=$2 g=$3; shift 3
  local cmd="$BIN serve --backend metal --model $M35 --max-batch 8 --served-model-name intarsia --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT"
  echo "## $tag $(date +%T)"
  env X=1 "$@" $cmd > $O/$tag-server.out 2> $O/$tag-server.err &
  local sp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 1024 --gen $g --reps 1 \
     --concurrency $conc --thinking off --label "mdec-w1-$tag" --recipe "$* $cmd" > $O/$tag-localeval.out 2> $O/$tag-localeval.err; echo "localeval exit $?")
  kill $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}
srv b-plain 1,4,8,16 256
srv b-host 8,16 256 INTARSIA_HOST_PROFILE=1
srv b-cb 8 128 INTARSIA_METAL_PROFILE=cb:$O/b-cb.jsonl
gzip -f $O/*.jsonl
echo "## done $(date +%T)"
