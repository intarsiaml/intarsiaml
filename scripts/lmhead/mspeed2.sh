#!/bin/zsh
# Lane lmhead-fp8-default, Metal decode sanity (endor, exclusive): the 27B default (recipe fp8 head, MTP k2 from the recipe) against INTARSIA_LMHEAD_QUANT=off,
# a fresh `serve` per visit, ABAB, localeval speed 1k / 4k, gen 128, 3 reps.
#   fleetq run --host endor --exclusive --mem 30 --expect 30 --label "..." -- zsh scripts/lmhead/mspeed2.sh <out dir>
set -u
R=${0:A:h:h:h}; O=${1:?out dir}; mkdir -p $O; BIN=$R/target/release/intarsia; PORT=8298
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
[[ -z "${INTARSIA_LMHEAD_QUANT:-}" ]] || { echo "INTARSIA_LMHEAD_QUANT is set"; exit 2; }
zsh $R/scripts/lmhead/mrebuild.sh | tee $O/rebuild.txt
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; n=0
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
SPID=""; trap '[[ -n "$SPID" ]] && kill $SPID 2>/dev/null' EXIT
for v in D O D O; do
  n=$((n+1)); envs=(); [[ $v == O ]] && envs=(INTARSIA_LMHEAD_QUANT=off)
  say "visit $n $v: starting server"
  env $envs $BIN serve --backend metal --model $M --served-model-name qwen38-27b --max-seq 8192 --profile-every 0 --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$n-$v.out 2> $O/serve-$n-$v.err &
  SPID=$!
  up=0; for i in {1..240}; do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 1; done
  [[ $up == 1 ]] || { say "visit $n $v: server failed"; SPID=""; continue; }
  grep -ah "lm_head quantised\|speculative decoding" $O/serve-$n-$v.err | head -2 | cut -c1-160 | tee -a $O/steps.log
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model qwen38-27b --thinking off --sizes 1024,4096 --gen 128 --reps 3 \
      --label "lmh-default-metal-$n-$v" --recipe "intarsia serve metal 27B default=$v (lane lmhead-fp8-default)" --wait-idle 60 ) > $O/speed-$n-$v.log 2>&1 < /dev/null
  say "item $n $v speed: exit $? $(grep -aoE 'results/[^ ]+' $O/speed-$n-$v.log | tail -1)"
  kill $SPID 2>/dev/null; for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""
  for i in {1..30}; do lsof -iTCP:$PORT -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done
done
say done
