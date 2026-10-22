#!/bin/zsh
# so lane: ONE exclusive endor slot. Qwen3-8B BF16, Metal.
#   1. profile: prof binary, INTARSIA_CONSTRAIN_PROFILE=1, fruits-40 speed x3 (per-phase ns on stderr)
#   2. A/B: base vs head, ROUNDS rounds, order alternates, fresh server per visit, speed x REPS (+ identity in rounds 1-2)
#   3. llama.cpp (BF16 GGUF) same schema and prompt, speed x ROUNDS visits... (one server, ROUNDS x REPS reps)
# fleetq run --host endor --exclusive --mem 24 --expect 45 --label "so: constrained decode profile + A/B" -- zsh run.sh <out dir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
OUT=${1:?out dir}
ROUNDS=${ROUNDS:-5}
REPS=${REPS:-1}
H=${0:A:h}
MODEL=${MODEL:-/Users/josef/models/Qwen3-8B}
GGUF=${GGUF:-/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf}
BASE=/opt/code/wt-so-base/target-metal/release/intarsia
PROF=/opt/code/wt-so-prof/target-metal/release/intarsia
HEAD=/opt/code/wt-so/target-metal/release/intarsia
PORT=8093
mkdir -p $OUT
for b in $BASE $PROF $HEAD; do [ -x $b ] || { echo "missing binary $b" >&2; exit 3; }; done
SPID=""
trap '[ -n "$SPID" ] && kill $SPID 2>/dev/null' EXIT
{ echo "date: $(date -Iseconds) fleetq job: $FLEETQ_JOB"; for b in $BASE $PROF $HEAD; do echo "$b $(md5 -q $b)"; done; sw_vers | tr '\n' ' '; echo; } > $OUT/versions.txt 2>&1

serve() { # label bin [env...]
  local label=$1 bin=$2; shift 2
  env INTARSIA_CPU_FREE_MEM_GB=60 "$@" $bin serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port $PORT \
    --served-model-name m --max-seq 4096 --profile-every 0 --no-supervisor --no-ui > $OUT/$label-server.out 2> $OUT/$label-server.err < /dev/null &
  SPID=$!
  for i in {1..240}; do
    curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && return 0
    kill -0 $SPID 2>/dev/null || break
    sleep 1
  done
  echo "SERVER FAILED $label" | tee -a $OUT/job.log
  return 1
}
stop() {
  [ -n "$SPID" ] && { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }
  SPID=""
  sleep 3
}
warm() {
  curl -sf http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
    -d '{"prompt":"Hello","max_tokens":16,"temperature":0}' > /dev/null < /dev/null
}
SS="uv run --script $H/ssbench.py"

echo "== profile $(date)" | tee -a $OUT/job.log
serve prof $PROF INTARSIA_CONSTRAIN_PROFILE=1 && {
  warm
  $SS speed --url http://127.0.0.1:$PORT --reps 3 --label prof --out $OUT/prof-speed.jsonl < /dev/null 2>&1 | tee -a $OUT/job.log
}
stop

for r in {1..$ROUNDS}; do
  if (( r % 2 )); then order=(base head); else order=(head base); fi
  for arm in $order; do
    echo "== round $r $arm $(date)" | tee -a $OUT/job.log
    bin=$BASE; [[ $arm == head ]] && bin=$HEAD
    serve $arm-r$r $bin || continue
    warm
    $SS speed --url http://127.0.0.1:$PORT --reps $REPS --label $arm-r$r --out $OUT/ab-speed.jsonl < /dev/null 2>&1 | tee -a $OUT/job.log
    (( r <= 2 )) && $SS identity --url http://127.0.0.1:$PORT --out $OUT/identity-$arm-r$r.json < /dev/null > $OUT/identity-$arm-r$r.log 2>&1
    stop
  done
done

echo "== llama.cpp $(date)" | tee -a $OUT/job.log
/opt/homebrew/bin/llama-server -m $GGUF -ngl 999 -fa on -c 8192 --cache-ram 0 --port 8091 > $OUT/llama-server.out 2> $OUT/llama-server.err < /dev/null &
SPID=$!
up=0
for i in {1..240}; do curl -sf http://127.0.0.1:8091/health > /dev/null 2>&1 && { up=1; break; }; sleep 1; done
if [[ $up == 1 ]]; then
  curl -sf http://127.0.0.1:8091/v1/completions -H 'content-type: application/json' -d '{"prompt":"Hello","max_tokens":16,"temperature":0}' > /dev/null < /dev/null
  $SS speed --url http://127.0.0.1:8091 --engine llama --reps $(( ROUNDS * REPS )) --label llama --out $OUT/llama-speed.jsonl < /dev/null 2>&1 | tee -a $OUT/job.log
fi
stop
echo "== done $(date)" | tee -a $OUT/job.log
