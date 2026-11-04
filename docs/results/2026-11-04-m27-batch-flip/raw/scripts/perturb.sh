#!/bin/zsh
# independent rounding perturbations of the state at the end of prefill; decode kernels are the same in all of them
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$PWD/.scratch/intarsia.head; O=$PWD/.scratch/logs/oracle; M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18477
export INTARSIA_CB_TRACE=1 INTARSIA_CB_NO_REUSE=1
serve() { local T=$1; shift
  $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@" > $O/p-$T-server.out 2> $O/p-$T-server.err < /dev/null &
  SP=$!
  local up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $T"; exit 1; }
}
stop() { kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null; }
trap 'stop' EXIT
R=$O/perturb.json
INTARSIA_METAL_MXFP4_NAX=0 serve nax0; python3 .scratch/perturb.py $PORT $R solo nax0 258 260 262 1028; stop
serve kvf32 --kv-dtype f32; python3 .scratch/perturb.py $PORT $R solo kvf32 258 260 262 1028; stop
for pl in 258 260 262 1028; do for t in 3 5 7; do
  serve tail$pl-$t --busy-chunk $((pl-t)); python3 .scratch/perturb.py $PORT $R behind tail$t $pl; stop
done; done
serve forced; python3 .scratch/perturb.py $PORT $R solo forced 258@100 258@104 260@138 260@197 262@188 1028@63 1028@129; stop
trap - EXIT
