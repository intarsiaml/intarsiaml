#!/bin/zsh
# Second half of the 8B control, cut to the prompts the f64 side finished: f32 kernels on 258 and 260.
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$PWD/.scratch/intarsia.main; O=$PWD/.scratch/logs/oracle; M=/Users/josef/models/Qwen3-8B; PORT=18480
$BIN serve --backend cpu --model $M --served-model-name intarsia --max-seq 2000 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/8b-f32-server.out 2> $O/8b-f32-server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..600}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; exit 1; }
python3 .scratch/control-8b.py $PORT $O/8b-f32.json 258,260
