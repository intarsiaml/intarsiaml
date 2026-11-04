#!/bin/zsh
# f32 CPU kernels free-running (a pure precision perturbation: no Metal, no GEMV / NAX split), then f64 reference teacher-forced at the rows it diverged before
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$PWD/.scratch/intarsia.head; O=$PWD/.scratch/logs/oracle; M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18476
export PROBE_TIMEOUT=3000
srv() { local T=$1; shift
  $BIN serve --backend cpu "$@" --model $M --served-model-name intarsia --max-seq 2000 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/$T-server.out 2> $O/$T-server.err < /dev/null &
  SP=$!
  local up=0; for i in {1..600}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $T"; exit 1; }
}
stop() { kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null; }
trap 'stop' EXIT
L=.scratch/logs/exp3/len.json
srv ref3 --reference
python3 .scratch/oracle-probe.py $PORT $O/ref-forced.json 260 2 $L solo 197
python3 .scratch/oracle-probe.py $PORT $O/ref-forced.json 262 2 $L solo 188
python3 .scratch/oracle-probe.py $PORT $O/ref-forced.json 1028 2 $L solo 63
stop
srv f32
for a in "258 108" "260 200" "1028 66"; do set -- ${=a}; python3 .scratch/oracle-probe.py $PORT $O/f32-free.json $1 $2; done
stop
trap - EXIT
