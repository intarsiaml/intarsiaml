#!/usr/bin/env bash
# Lane bm, N=1 default: CUDA gates on spark1, Mac side, inside a fleetq slot (FLEETQ_JOB set):
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 30 --label "bm-default: CUDA gates" -- scripts/bm/gates-default.sh
# (1) plan show (single-sequence, 35B) branch vs main d40d93896 (#310, packed-prefill fix): byte-identical. (2) greedy ids of 6 concurrent prompts, a `--max-batch 16`
# server with NO --spec and NO INTARSIA_BATCHED_MTP (the recipe default), vs the solo spec-off server; packed prefill on (the default, fixed by #310) and off.
# (3) the startup line names N=1 and /metrics shows drafting. (4) INTARSIA_BATCHED_MTP=0 turns it off (no drafting).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=spark1; HN=spark-4e77; PORT=8195
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
O=$REPO/docs/results/2026-11-05-batched-mtp/raw/default-cuda; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
B='~/intarsiaml-bm/bin'
rsh() { ssh -n -o BatchMode=yes $HOST "$@"; }
echo "== plan show $(date +%T)"
for b in main310 def; do rsh "$B/$b plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json" > $O/plan-$b.json 2> $O/plan-$b.err; done
cmp $O/plan-main310.json $O/plan-def.json && echo "plan show json: branch == main d40d93896 (byte-identical)" | tee $O/plan.txt || { echo "plan show json: DIFFERS" | tee $O/plan.txt; diff $O/plan-main310.json $O/plan-def.json | head -20; }
diff $O/plan-main310.err $O/plan-def.err > $O/plan-err.diff && echo "plan show notes: identical" | tee -a $O/plan.txt || { echo "plan show notes differ:" | tee -a $O/plan.txt; cat $O/plan-err.diff | tee -a $O/plan.txt; }
SPID=""
stop() { [[ -n "$SPID" ]] && { rsh "kill $SPID; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID"; SPID=""; }; }
trap stop EXIT
rsh "mkdir -p ~/intarsiaml-bm/out"
ids() { # ids <name> <env> <conc> <args...>
  local name=$1 envs=$2 conc=$3; shift 3
  echo "== ids $name $(date +%T)"
  SPID=$(rsh "nohup env -u INTARSIA_BATCHED_MTP $envs $B/def serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name m --max-seq 8192 --mem-budget 40 --profile-every 0 $* > ~/intarsiaml-bm/out/def-$name.log 2>&1 < /dev/null & echo \$!")
  local up=0; for i in $(seq 1 150); do curl -sf http://$HN:$PORT/health >/dev/null && { up=1; break; }; sleep 2; done
  if [[ $up == 1 ]]; then python3 $REPO/scripts/bm/ids.py http://$HN:$PORT/v1 m $O/ids-$name.json $conc
    curl -sf http://$HN:$PORT/metrics | grep -E '^intarsia_spec_(proposed|accepted)_tokens_total' | tee $O/ids-$name.metrics
  else echo "server $name did not come up"; fi
  rsh "cat ~/intarsiaml-bm/out/def-$name.log" > $O/serve-$name.log
  grep -E "batched MTP|--spec mtp|spec" $O/serve-$name.log | head -3
  stop
}
ids solo-off "" 1 --max-batch 1 --spec off
ids default16-nopack "INTARSIA_PACK_PREFILL=0" 6 --max-batch 16
ids default16 "" 6 --max-batch 16
ids env0-16 "INTARSIA_BATCHED_MTP=0" 6 --max-batch 16
for n in default16-nopack default16 env0-16; do
  python3 - "$O/ids-solo-off.json" "$O/ids-$n.json" <<'PY' | tee -a $O/ids.txt
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = [x == y for x, y in zip(a, b)]
print(sys.argv[2].split('/')[-1], "identical per sequence:", same, "ALL" if all(same) and len(a) == len(b) else "MISMATCH")
PY
done
echo "== done $(date +%T)"
