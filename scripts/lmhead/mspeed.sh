#!/bin/zsh
# Lane lmhead-fp8, Metal speed (endor, Qwen3.8-27B MXFP4 W4A16), ONE exclusive slot, a fresh `serve` per visit, ABAB:
#   fleetq run --host endor --exclusive --mem 30 --expect 60 --label "..." -- zsh scripts/lmhead/mspeed.sh <binary> <out dir>
# VISITS (default) = "off.off fp8.off off.off fp8.off off.mtp fp8.mtp off.mtp fp8.mtp mxfp4.off off.off mxfp4.mtp": Q.SPEC, Q = INTARSIA_LMHEAD_QUANT
# (off = unset), SPEC off = --spec off, mtp = the recipe's default (MTP k2, draft head 65536 rows). localeval speed 1k/4k/16k, gen 128, 3 reps.
set -u
BIN=${1:?binary}; O=${2:?out dir}; mkdir -p $O
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
zsh ${0:A:h}/mrebuild.sh | tee $O/rebuild.txt
M=${M:-/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4}; PORT=8298; n=0
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
SPID=""; trap '[[ -n "$SPID" ]] && kill $SPID 2>/dev/null' EXIT
for v in ${=VISITS:-off.off fp8.off off.off fp8.off off.mtp fp8.mtp off.mtp fp8.mtp mxfp4.off off.off mxfp4.mtp}; do
  n=$((n+1)); q=${v%.*}; sp=${v#*.}; extra=(); [[ $sp == off ]] && extra=(--spec off)
  envs=(); [[ $q != off ]] && envs=(INTARSIA_LMHEAD_QUANT=$q)
  say "visit $n $v: starting server"
  env $envs $BIN serve --backend metal --model $M --served-model-name qwen38-27b --max-seq 20480 --profile-every 0 --no-supervisor $extra --host 127.0.0.1 --port $PORT > $O/serve-$n-$v.out 2> $O/serve-$n-$v.err &
  SPID=$!
  up=0; for i in {1..240}; do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 1; done
  [[ $up == 1 ]] || { say "visit $n $v: server failed"; SPID=""; continue; }
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model qwen38-27b --thinking off --sizes 1024,4096,16384 --gen 128 --reps 3 \
      --label "lmh-metal-$n-$v" --recipe "intarsia serve metal q=$q spec=$sp (lane lmhead-fp8)" --wait-idle 60 ) > $O/speed-$n-$v.log 2>&1 < /dev/null
  say "item $n $v speed: exit $? $(grep -aoE 'results/[^ ]+' $O/speed-$n-$v.log | tail -1)"
  kill $SPID 2>/dev/null; for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""
  for i in {1..30}; do lsof -iTCP:$PORT -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done
done
say done
