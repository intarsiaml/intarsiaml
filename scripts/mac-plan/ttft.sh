#!/bin/zsh
# TTFT probe: 35B served, MTP default vs --spec off, one 4k prompt x3 fresh nonces per arm. Inside a fleetq slot.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mac-plan; BIN=$W/target/release/intarsia; R=$W/scripts/mac-plan/raw/ttft; mkdir -p $R
M=${MODEL:-$HOME/models/Qwen3.6-35B-A3B-NVFP4}; PF=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-${CTX:-4096}.txt; PORT=18612
for arm in "$@"; do
  tag=${arm%%:*}; rest=${arm#*:}; [ "$rest" = "$arm" ] && rest=""
  envs=${rest%%:*}; flags=${rest#*:}; [ "$flags" = "$rest" ] && flags=""
  PE=(); [[ $tag == *cb ]] && PE=(INTARSIA_METAL_PROFILE=cb:$R/$tag.jsonl)
  env X=1 $PE ${=envs//,/ } $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT ${=flags} > $R/$tag.out 2> $R/$tag.err < /dev/null &
  TP=$!; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  echo "## $tag [$envs] [$flags]"
  for n in a b c; do python3 $W/scripts/mac-plan/ttft.py http://127.0.0.1:$PORT/v1 $PF $tag$n; done
  kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
  grep -a "spec:" $R/$tag.err | tail -1
done
