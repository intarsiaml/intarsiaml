#!/bin/zsh
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$PWD/.scratch/intarsia.head; O=$PWD/.scratch/logs/oracle; M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18476
$BIN serve --backend cpu --reference --model $M --served-model-name intarsia --max-seq 2000 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/ref-server.out 2> $O/ref-server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..600}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; tail -5 $O/ref-server.err; exit 1; }
python3 .scratch/oracle-probe.py $PORT $O/ref-time.json 258 3 || echo PROBE-TIMEOUT
