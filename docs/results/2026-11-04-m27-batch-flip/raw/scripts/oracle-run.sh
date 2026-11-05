#!/bin/zsh
# free-running f64 reference from each prompt to just past its swing row
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$PWD/.scratch/intarsia.head; O=$PWD/.scratch/logs/oracle; M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18476
export PROBE_TIMEOUT=3600
$BIN serve --backend cpu --reference --model $M --served-model-name intarsia --max-seq 2000 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/ref2-server.out 2> $O/ref2-server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..600}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; exit 1; }
for a in "258 108" "260 200" "262 192" "1028 66"; do set -- ${=a}; python3 .scratch/oracle-probe.py $PORT $O/ref-free.json $1 $2; done
