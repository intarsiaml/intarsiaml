#!/bin/zsh
# TTFT probe (lane mtt; from scripts/mac-plan/ttft.sh): one fresh `serve` per arm, /v1/completions, a prompt with a fresh nonce per call.
# usage (inside a fleetq slot): OUT=<raw dir> [CTX=4096] [CALLS=3] [MODEL=...] [GEN=48] ttft.sh ARM...
#   COOL=N waits up to N s for thermal pressure level 0 before each arm (the level is printed per arm either way).
#   ARM = tag[:ENV=V,ENV=V[:serve flags]]; tags ending in "cb" get a command-buffer trace; MAXSEQ=N and SAMPLE=1 in the env list
#   and LOADGEN=N are taken by the script (--max-seq; `sample` of the server during the second call; N busy CPU threads beside the calls).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${W:-/opt/code/intarsiaml/.worktrees/mtt}; BIN=${BIN:-$W/target/metal/release/intarsia}; R=${OUT:?OUT=raw dir}; mkdir -p $R
M=${MODEL:-$HOME/models/Qwen3.6-35B-A3B-NVFP4}; PF=${PROMPT:-$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-${CTX:-4096}.txt}; PORT=${PORT:-18613}
for arm in "$@"; do
  tag=${arm%%:*}; rest=${arm#*:}; [ "$rest" = "$arm" ] && rest=""
  envs=${rest%%:*}; flags=${rest#*:}; [ "$flags" = "$rest" ] && flags=""
  ms=40000; smp=0; EV=()
  lg=0
  for e in ${=envs//,/ }; do case $e in MAXSEQ=*) ms=${e#*=};; SAMPLE=*) smp=${e#*=};; LOADGEN=*) lg=${e#*=};; *) EV+=($e);; esac; done
  # COOL=N: wait up to N s for the box to report no thermal pressure before the arm starts (a hot box throttles the GPU by 1.2..1.45x)
  if [ -n "${COOL:-}" ]; then c0=$(date +%s); while [ "$(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}')" != 0 ] && [ $(( $(date +%s) - c0 )) -lt $COOL ]; do sleep 5; done; fi
  PE=(); [[ $tag == *cb ]] && PE=(INTARSIA_METAL_PROFILE=cb:$R/$tag.jsonl)
  env X=1 $PE $EV $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq $ms --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT ${=flags} > $R/$tag.out 2> $R/$tag.err < /dev/null &
  TP=$!; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  # LOADGEN=N: N busy CPU threads beside the arm's calls (what a build or test run of another job does to the probe)
  LG=(); for j in {1..$lg}; do [ $lg -gt 0 ] && { perl -e 'alarm 120; 1 while 1' & LG+=($!); }; done
  echo "## $tag [$envs] [$flags] $(date +%T) thermal $(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | tr -d '\n') load $(grep -a '^loaded' $R/$tag.err | sed 's/.* in //;s/,.*//')"
  i=0
  for n in a b c d e f g h; do
    i=$((i+1)); [ $i -gt ${CALLS:-3} ] && break
    [ $smp = 1 ] && [ $i = 2 ] && { sample $TP 1 1 -file $R/$tag.sample.txt >/dev/null 2>&1 & }
    python3 $W/scripts/mtt/ttft.py http://127.0.0.1:$PORT/v1 $PF $tag$n${PLAIN:+}${PLAIN:--$RANDOM} ${GEN:-48}
  done
  wait %sample 2>/dev/null
  [ $#LG -gt 0 ] && kill $LG 2>/dev/null
  kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
  grep -a "spec:" $R/$tag.err | tail -1
done
