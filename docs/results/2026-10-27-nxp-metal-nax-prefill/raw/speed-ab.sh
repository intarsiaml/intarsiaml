#!/bin/bash
# G4: interleaved A/B localeval speed (35B, max-batch 1), ROUNDS rounds of "arm1 arm2" (order flips each round), inside ONE fleetq slot.
# usage: speed-ab.sh OUT ROUNDS label=BIN[,ENV=V...] label=BIN ...
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
OUT=$1; R=$2; shift 2; mkdir -p "$OUT"; PORT=18435
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
arm() { # round label bin envs
  local r=$1 label=$2 bin=$3 envs=$4 tag=$1-$2
  echo "##### $tag $(date -Iseconds) rustc+cargo before: $(pgrep -x rustc | wc -l)+$(pgrep -x cargo | wc -l)"
  local cmd="$bin serve --backend metal --model $M35 --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT"
  env ${envs//,/ } $cmd > "$OUT/$tag-server.out" 2> "$OUT/$tag-server.err" &
  local sp=$!
  (while true; do echo "$(date +%T) $(ps -axo pid,%cpu,comm -r | sed -n 2,4p | tr -s " " | tr "\n" ";") rustc=$(pgrep -x rustc | wc -l)"; sleep 4; done) > "$OUT/$tag-load.log" 2>&1 &
  local lp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  echo "up after ${i}s"
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 1024,4096,16384 --gen 256 --reps 3 \
     --thinking off --label "nxp-$tag" --recipe "$envs $cmd" > "$OUT/$tag-localeval.out" 2> "$OUT/$tag-localeval.err"; echo "localeval exit $?")
  echo "rustc+cargo after: $(pgrep -x rustc | wc -l)+$(pgrep -x cargo | wc -l)  $(date -Iseconds)"
  kill $lp 2>/dev/null; kill $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}
md5 -q $(for a in "$@"; do b=${a#*=}; echo ${b%%,*}; done) > "$OUT/versions.txt"
for r in $(seq 1 $R); do
  arms=("$@"); [ $((r % 2)) = 0 ] && arms=("${arms[@]}") && arms=($(printf '%s\n' "$@" | tail -r))
  for a in "${arms[@]}"; do label=${a%%=*}; rest=${a#*=}; bin=${rest%%,*}; envs=""; [ "$rest" != "$bin" ] && envs=${rest#*,}; arm $r $label $bin "$envs"; done
done
