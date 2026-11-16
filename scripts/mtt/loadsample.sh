#!/bin/zsh
# `sample` a `serve` start-up (lane mtt: what the MTP default adds to the load). usage (fleetq slot): OUT=dir loadsample.sh tag [serve flags]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${W:-/opt/code/intarsiaml/.worktrees/mtt}; BIN=${BIN:-$W/target/metal/release/intarsia}; R=${OUT:?}; mkdir -p $R; PORT=${PORT:-18613}
M=${MODEL:-$HOME/models/Qwen3.6-35B-A3B-NVFP4}; tag=$1; shift
$BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@" > $R/$tag.out 2> $R/$tag.err < /dev/null &
TP=$!; sample $TP ${SECS:-12} 2 -file $R/$tag.sample.txt > /dev/null 2>&1 &
for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 0.2; done
wait %2 2>/dev/null; grep -a "^loaded" $R/$tag.err
kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
