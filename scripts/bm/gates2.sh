#!/usr/bin/env bash
# Lane bm: CUDA gates on spark1, Mac side, inside a fleetq slot (FLEETQ_JOB set):
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 60 --label "bm: CUDA gates" -- scripts/bm/gates.sh
# (1) plan show, branch vs main, no opt-in: byte-identical json; and with the opt-in (memory note). (2) kernels test --quick.
# (3) greedy ids: solo spec-off server vs batched-MTP servers, phase A (solo passes) and B (INTARSIA_BATCHED_VERIFY=1), 6 concurrent
# prompts of different lengths and max_tokens.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=spark1; HN=spark-4e77; PORT=8194
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
O=$REPO/docs/results/2026-11-05-batched-mtp/raw/gates-cuda2; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
B='~/intarsiaml-bm/bin'
rsh() { ssh -n -o BatchMode=yes $HOST "$@"; }
echo "== plan show $(date +%T)"
for b in main bm; do rsh "$B/$b plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json" > $O/plan-$b.json 2> $O/plan-$b.err; done
cmp $O/plan-main.json $O/plan-bm.json && echo "plan show: branch == main (byte-identical)" | tee $O/plan.txt || echo "plan show: DIFFERS" | tee $O/plan.txt
rsh "INTARSIA_BATCHED_MTP=4 $B/bm plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --spec mtp --json" > $O/plan-bm-mtp.json 2> $O/plan-bm-mtp.err
echo "== kernels test --quick $(date +%T)"
rsh "$B/bm kernels test --backend cuda --quick" > $O/kernels.log 2>&1; echo "kernels exit $?"; grep -cE "NOT ADMITTED|FAILED" $O/kernels.log; tail -3 $O/kernels.log
SPID=""
stop() { [[ -n "$SPID" ]] && { rsh "kill $SPID; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID"; SPID=""; }; }
trap stop EXIT
serve() { # serve <name> <env> <args...>
  local name=$1 envs=$2; shift 2
  SPID=$(rsh "nohup env $envs $B/bm serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name m --max-seq 8192 --mem-budget 40 --profile-every 0 $* > ~/intarsiaml-bm/out/gate2-$name.log 2>&1 < /dev/null & echo \$!")
  for i in $(seq 1 150); do curl -sf http://$HN:$PORT/health >/dev/null && return 0; sleep 2; done
  echo "server $name did not come up"; rsh "tail -5 ~/intarsiaml-bm/out/gate2-$name.log"; return 1
}
ids() { # ids <name> <env> <conc> <args...>
  local name=$1 envs=$2 conc=$3; shift 3
  echo "== ids $name $(date +%T)"
  serve $name "$envs" "$@" && python3 $REPO/scripts/bm/ids.py http://$HN:$PORT/v1 m $O/ids-$name.json $conc
  curl -sf http://$HN:$PORT/metrics | grep -E '^intarsia_spec_(proposed|accepted)_tokens_total' | tee $O/ids-$name.metrics
  rsh "cat ~/intarsiaml-bm/out/gate2-$name.log" > $O/serve-$name.log
  stop
}
rsh "mkdir -p ~/intarsiaml-bm/out"
ids solo-off "" 1 --max-batch 1 --spec off
ids bmtp4 "INTARSIA_BATCHED_MTP=4" 6 --max-batch 4 --spec mtp --spec-k 2 --spec-off-at 5
ids bmtp1 "INTARSIA_BATCHED_MTP=1" 6 --max-batch 16 --spec mtp --spec-k 2
ids plain16 "" 6 --max-batch 16
ids bver5 "INTARSIA_BATCHED_MTP=5 INTARSIA_BATCHED_VERIFY=1" 6 --max-batch 8 --spec mtp --spec-k 2 --spec-off-at 9
ids bver8k3 "INTARSIA_BATCHED_MTP=8 INTARSIA_BATCHED_VERIFY=1" 6 --max-batch 16 --spec mtp --spec-k 3 --spec-off-at 9
for n in bmtp4 bmtp1 plain16 bver5 bver8k3; do
  python3 - "$O/ids-solo-off.json" "$O/ids-$n.json" <<'PY' | tee -a $O/ids.txt
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = [x == y for x, y in zip(a, b)]
print(sys.argv[2].split('/')[-1], "identical per sequence:", same, "ALL" if all(same) and len(a) == len(b) else "MISMATCH")
PY
done
echo "== done $(date +%T)"
