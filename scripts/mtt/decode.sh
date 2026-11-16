#!/bin/zsh
# Lane mtt speed rule: decode tok/s (localeval speed, 4k and 16k prompts, 256 forced tokens), MTP default, lazy admission on vs off
# (INTARSIA_METAL_MTP_TTFT_FIX=0), fresh server per arm, A B B A. usage (fleetq slot): OUT=dir decode.sh MODEL_DIR TAG
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${W:-/opt/code/intarsiaml/.worktrees/mtt}; BIN=${BIN:-$W/target/metal/release/intarsia}; R=${OUT:?}; mkdir -p $R; PORT=${PORT:-18613}; M=$1; T=$2
for arm in on1 old1 old2 on2; do
  c0=$(date +%s); while [ "$(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}')" != 0 ] && [ $(( $(date +%s) - c0 )) -lt ${COOL:-90} ]; do sleep 5; done
  EV=(); [[ $arm == old* ]] && EV=(INTARSIA_METAL_MTP_TTFT_FIX=0)
  CMD=($BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT)
  env X=1 $EV $CMD > $R/$T-$arm.out 2> $R/$T-$arm.err < /dev/null &
  TP=$!; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  echo "## $T $arm $(date +%T) thermal $(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}') load $(grep -a '^loaded' $R/$T-$arm.err | sed 's/.* in //;s/,.*//')"
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 4096,16384 --gen 256 --reps ${REPS:-2} --thinking off \
     --label "mtt-$T-$arm" --recipe "$EV ${CMD[*]}" > $R/$T-$arm.localeval.out 2> $R/$T-$arm.localeval.err; echo "localeval exit $?")
  kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
  grep -a "^| *[0-9]" $R/$T-$arm.localeval.out
done
