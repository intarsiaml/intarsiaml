#!/bin/zsh
# Non-recurrent control: Qwen3-8B (dense attention layers, no GatedDeltaNet), the same f32-vs-f64 CPU comparison, 200 ids free-running.
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$PWD/.scratch/intarsia.main; O=$PWD/.scratch/logs/oracle; M=/Users/josef/models/Qwen3-8B; PORT=18480
for t in f64 f32; do
  [ $t = f64 ] && A=(--reference) || A=()
  $BIN serve --backend cpu $A --model $M --served-model-name intarsia --max-seq 2000 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/8b-$t-server.out 2> $O/8b-$t-server.err < /dev/null &
  SP=$!
  trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
  up=0; for i in {1..600}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $t"; exit 1; }
  python3 .scratch/control-8b.py $PORT $O/8b-$t.json 258,260,262,1028
  kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
done
trap - EXIT
