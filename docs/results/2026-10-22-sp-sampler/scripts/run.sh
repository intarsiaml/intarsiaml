#!/bin/zsh
# sp lane: ONE exclusive endor slot. Qwen3-8B BF16, Metal.
# Visits base, head, head, base (fresh server each): ssbench speed x REPS; seeded sampled token ids in round 1.
# fleetq run --host endor --exclusive --mem 24 --label "sp: sampler A/B + seeded identity" -- zsh run.sh <out dir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
OUT=${1:?out dir}; REPS=${REPS:-2}
H=${0:A:h}
MODEL=${MODEL:-/Users/josef/models/Qwen3-8B}
BASE=/opt/code/wt-sp-base/target-metal/release/intarsia
HEAD=${0:A:h:h:h:h:h}/target-metal/release/intarsia
PORT=8094
mkdir -p $OUT
for b in $BASE $HEAD; do [ -x $b ] || { echo "missing binary $b" >&2; exit 3; }; done
SPID=""
trap '[ -n "$SPID" ] && kill $SPID 2>/dev/null' EXIT
{ echo "date: $(date -Iseconds) fleetq job: $FLEETQ_JOB"; for b in $BASE $HEAD; do echo "$b $(md5 -q $b)"; done; sw_vers | tr '\n' ' '; echo; } > $OUT/versions.txt 2>&1
serve() { # label bin
  local label=$1 bin=$2
  env INTARSIA_CPU_FREE_MEM_GB=60 $bin serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port $PORT \
    --served-model-name m --max-seq 4096 --profile-every 0 --no-supervisor --no-ui > $OUT/$label-server.out 2> $OUT/$label-server.err < /dev/null &
  SPID=$!
  for i in {1..240}; do
    curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && return 0
    kill -0 $SPID 2>/dev/null || break
    sleep 1
  done
  echo "SERVER FAILED $label" | tee -a $OUT/job.log; return 1
}
stop() { [ -n "$SPID" ] && { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""; sleep 3; } }
SS=(uv run --script $H/ssbench.py)
r=0
for arm in base head head base; do
  r=$((r + 1))
  echo "== visit $r $arm $(date)" | tee -a $OUT/job.log
  bin=$BASE; [[ $arm == head ]] && bin=$HEAD
  serve $arm-v$r $bin || continue
  curl -sf http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' -d '{"prompt":"Hello","max_tokens":16,"temperature":0}' > /dev/null < /dev/null
  ${SS[@]} speed --url http://127.0.0.1:$PORT --reps $REPS --label $arm-v$r --out $OUT/ab-speed.jsonl < /dev/null 2>&1 | tee -a $OUT/job.log
  (( r <= 2 )) && python3 $H/seeded_ids.py --url http://127.0.0.1:$PORT --out $OUT/ids-$arm.json < /dev/null > $OUT/ids-$arm.log 2>&1
  stop
done
python3 - $OUT/ids-base.json $OUT/ids-head.json <<'PY' | tee -a $OUT/job.log
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
diff = [k for k in a if a[k] != b.get(k)]
print(f"seeded identity: {len(a)} cases, {sum(len(v) for v in a.values())} tokens, {len(diff)} differ {diff}")
PY
echo "== done $(date)" | tee -a $OUT/job.log
