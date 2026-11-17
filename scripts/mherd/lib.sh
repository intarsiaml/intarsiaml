#!/bin/zsh
# Lane mherd (Metal): shared helpers. Source inside a fleetq slot on endor. Binary: target/release/intarsia of this worktree (== origin/main 36f21e2e).
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mherd; BIN=$W/target/release/intarsia; S=$W/scripts/mherd; R=$W/docs/results/2026-11-17-mherd/raw
MODEL=${MODEL:-35b}
case $MODEL in 27b) M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4 ;; 35b) M=$HOME/models/Qwen3.6-35B-A3B-NVFP4 ;; esac
PORT=${PORT:-18690}; URL=http://127.0.0.1:$PORT
th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
cool() { local w=0; while [ "$(th)" != 0 ] && [ $w -lt ${COOL:-90} ]; do sleep 5; w=$((w+5)); done; echo "cooled ${w}s thermal $(th)"; }
# start <dir> <env-assignments or -> <serve args...>
start() { local D=$1 E=$2; shift 2; mkdir -p $D
  local EV=(); [ "$E" != "-" ] && EV=(${=E})
  env INTARSIA_CB_NO_REUSE=1 $EV $BIN serve --backend metal --model $M --served-model-name intarsia --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT "$@" > $D/server.out 2> $D/server.err < /dev/null &
  TP=$!
  for i in {1..400}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && return 0; kill -0 $TP 2>/dev/null || break; sleep 1; done
  echo "SERVER FAILED $D: $(tail -3 $D/server.err)"; kill $TP 2>/dev/null; return 1
}
stop() { curl -sf $URL/metrics > $1/metrics.txt 2>/dev/null; kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; gzip -f $1/server.err; }
