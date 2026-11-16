#!/bin/zsh
# Load the GPU for SECS seconds (35B `--spec off` server, 32k-prompt requests back to back), to reproduce the box state a long
# sweep leaves behind. Inside a fleetq slot. usage: OUT=<raw dir> heat.sh SECS
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${W:-/opt/code/intarsiaml/.worktrees/mtt}; BIN=${BIN:-$W/target/release/intarsia}; R=${OUT:?}; mkdir -p $R; PORT=${PORT:-18613}
M=${MODEL:-$HOME/models/Qwen3.6-35B-A3B-NVFP4}; PF=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-32768.txt
$BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT --spec off > $R/heat.out 2> $R/heat.err < /dev/null &
TP=$!; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
t0=$(date +%s); i=0
while [ $(( $(date +%s) - t0 )) -lt ${1:-120} ]; do i=$((i+1)); python3 $W/scripts/mtt/ttft.py http://127.0.0.1:$PORT/v1 $PF heat$i-$RANDOM 256; done
echo "heat: $i requests in $(( $(date +%s) - t0 )) s; thermal $(notifyutil -g com.apple.system.thermalpressurelevel 2>&1)"
kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
