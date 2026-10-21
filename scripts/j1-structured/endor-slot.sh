#!/bin/bash
# J1 structured output on endor (Metal, Qwen3-8B BF16), base vs head. Two fleetq jobs:
#   fleetq run --host endor --mem 8 --label "J1 build (metal base+head)" -- \
#     bash scripts/j1-structured/endor-slot.sh build OUT BASE HEAD
#   fleetq run --host endor --exclusive --mem 24 --label "J1 e2e" -- \
#     bash scripts/j1-structured/endor-slot.sh run OUT BASE HEAD
# `run` builds too when the binaries are missing; `serve` is `run` without step 1. In order:
#   1. Metal no-op (docs/results/2026-10-21-m1-qwen38-drill/raw/metal-noop.sh): kernels test, plan, KL21 short + long,
#      256 greedy ids (short, 4k) base vs head; head `cargo test --release --workspace`.
#   2. head server: e2e.py (50 requests, sequential, + constrained decode speed), e2e.py --concurrency 4,
#      localeval GSM8K n=50 --thinking off, default (json_schema) and --structured prompt.
#   3. unconstrained overhead: `intarsia bench speed --at-most`, fresh serve per arm, A/B x3 interleaved (round 2
#      reversed), plus the HTTP greedy ids of one chat request per arm (return_token_ids).
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
MODE=$1 O=$2 BASE=$3 HEAD=$4
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
P=$R/docs/results/raw/2026-10-14-metal-phase-b
PORT=8107
NAME=qwen3-8b
SPID=
mkdir -p "$O"
trap '[ -n "$SPID" ] && kill $SPID 2>/dev/null; wait 2>/dev/null' EXIT

build() {
  for v in base:$BASE head:$HEAD; do
    n=${v%%:*}; rev=${v#*:}
    if [ -x "$O/target-$n/release/intarsia" ] && [ "$(cat "$O/$n.rev" 2>/dev/null)" = "$rev" ]; then
      echo "have $n $rev"; continue
    fi
    rm -rf "$O/$n"; mkdir -p "$O/$n"; git -C "$R" archive "$rev" | tar -x -C "$O/$n"
    (cd "$O/$n" && CARGO_TARGET_DIR=$O/target-$n cargo build --release -q -p intarsia-cli --features metal) \
      || { echo "build $n FAILED"; exit 2; }
    echo "$rev" > "$O/$n.rev"; echo "built $n $rev $(date +%T)"
  done
}

serve() { # arm tag [extra serve flags]
  local arm=$1 tag=$2; shift 2
  "$O/target-$arm/release/intarsia" serve --model $M --backend metal --max-seq 8192 --profile-every 0 --no-supervisor \
    --no-ui --host 127.0.0.1 --port $PORT --served-model-name $NAME "$@" > "$O/serve-$arm-$tag.log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 240); do
    curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && { echo "serve $arm $tag up $(date +%T)"; return 0; }
    kill -0 $SPID 2>/dev/null || break
    sleep 1
  done
  echo "serve $arm $tag did not come up"; tail -20 "$O/serve-$arm-$tag.log"; return 1
}

unserve() {
  [ -n "$SPID" ] && { kill $SPID; wait $SPID 2>/dev/null; }
  SPID=
  sleep 2
}

http_ids() { # arm tag: greedy ids of one unconstrained chat request over HTTP
  curl -sf http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' -d '{"messages":
    [{"role": "user", "content": "Write a short story about a lighthouse keeper and a robot."}], "max_tokens": 256,
    "temperature": 0, "return_token_ids": true, "chat_template_kwargs": {"enable_thinking": false}}' \
    | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["token_ids"])' > "$O/http-ids-$1-$2.txt"
  echo "http ids $1 $2: $(md5 -q "$O/http-ids-$1-$2.txt" | cut -c1-8) n=$(tr ',' '\n' < "$O/http-ids-$1-$2.txt" | wc -l)"
}

case "$MODE" in
build) build ;;
run|serve)
  date +%T
  build
  if [ "$MODE" = run ]; then
  # 1. no-op
  for n in base head; do
    "$O/target-$n/release/intarsia" kernels test --backend metal > "$O/$n-kernels.out" 2>&1; echo "$n kernels rc=$?"
  done
  for n in base head; do
    B=$O/target-$n/release/intarsia
    $B plan show --model $M --backend metal > "$O/$n-plan.out" 2> "$O/$n-plan.err"; echo "$n plan rc=$?"
    $B accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 \
      > "$O/$n-acc-short.out" 2> "$O/$n-acc-short.err"; echo "$n accshort rc=$?"
    $B accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 \
      > "$O/$n-acc-long.out" 2> "$O/$n-acc-long.err"; echo "$n acclong rc=$?"
    for p in short 4k; do
      $B generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
        --prompt "$(cat $P/prompt-$p.txt)" > "$O/$n-gen-$p.out" 2> "$O/$n-gen-$p.err"; echo "$n gen-$p rc=$?"
    done
  done
  # own target dir: a test build in target-head relinks release/intarsia without the metal feature
  (cd "$O/head" && CARGO_TARGET_DIR=$O/target-test cargo test --release --workspace > "$O/head-cargo-test.log" 2>&1)
  echo "head cargo test rc=$? $(date +%T)"
  fi

  # 2. structured output on the head server
  if serve head e2e; then
    uv run --script "$R/scripts/j1-structured/e2e.py" --url http://127.0.0.1:$PORT --out "$O/e2e-seq.json" \
      --constrained-speed > "$O/e2e-seq.log" 2>&1; echo "e2e seq rc=$? $(tail -1 "$O/e2e-seq.log")"
    # mixed batch: one unconstrained greedy request alone, then while 3 constrained requests decode beside it; its ids
    # must not change (device argmax per row) and its wall time is reported
    t0=$(python3 -I -c 'import time; print(time.time())'); http_ids head solo
    echo "solo wall $(python3 -I -c "import time; print(round(time.time() - $t0, 2))") s"
    uv run --script "$R/scripts/j1-structured/e2e.py" --url http://127.0.0.1:$PORT --out "$O/e2e-c4.json" \
      --concurrency 4 > "$O/e2e-c4.log" 2>&1 &
    E2E=$!
    sleep 8
    t0=$(python3 -I -c 'import time; print(time.time())'); http_ids head mixed
    echo "mixed wall $(python3 -I -c "import time; print(round(time.time() - $t0, 2))") s"
    cmp -s "$O/http-ids-head-solo.txt" "$O/http-ids-head-mixed.txt" && echo "mixed-batch ids: identical" || echo "mixed-batch ids: DIFFER"
    wait $E2E; echo "e2e c4 rc=$? $(tail -1 "$O/e2e-c4.log")"
    for mode in json_schema prompt; do
      (cd /opt/code/deepeval && uv run localeval quality --url http://127.0.0.1:$PORT/v1 --model $NAME --bench gsm8k \
        --limit 50 --thinking off --structured $mode --label "j1-intarsia-metal-$mode") > "$O/gsm8k-$mode.log" 2>&1
      echo "gsm8k $mode rc=$? $(date +%T)"
    done
    unserve
  fi

  # 3. unconstrained overhead, A/B x3 interleaved
  mkdir -p "$O/ab"
  for r in 1 2 3; do
    arms="base head"; [ $r = 2 ] && arms="head base"
    for arm in $arms; do
      serve $arm r$r || continue
      "$O/target-head/release/intarsia" bench speed --url http://127.0.0.1:$PORT --model $NAME --sizes 128,1024,4096 \
        --gen 128 --reps 2 --warmup 1 --at-most --thinking off --label "j1-$arm-r$r" --out-dir "$O/ab" \
        > "$O/ab/j1-$arm-r$r.log" 2>&1; echo "bench $arm r$r rc=$?"
      http_ids $arm r$r
      unserve
    done
  done

  echo "== summary"
  cd "$O"
  for f in plan.out acc-short.out acc-long.out gen-short.out gen-4k.out; do
    cmp -s base-$f head-$f && echo "$f: identical" || echo "$f: DIFFERS"
  done
  for p in short 4k; do
    b=$(grep '^ids:' base-gen-$p.err | md5 -q); h=$(grep '^ids:' head-gen-$p.err | md5 -q)
    [ "$b" = "$h" ] && echo "ids $p: identical (${b:0:8})" || echo "ids $p: DIFFER"
  done
  md5 -q http-ids-*.txt | sort | uniq -c | sed 's/^/http ids md5 count: /'
  tail -2 base-kernels.out; tail -2 head-kernels.out
  grep "test result" head-cargo-test.log | awk '{p+=$4; f+=$6} END {print "cargo test: passed", p, "failed", f}'
  tail -1 e2e-seq.log; tail -1 e2e-c4.log
  cmp -s http-ids-head-solo.txt http-ids-head-mixed.txt && echo "mixed-batch ids: identical" || echo "mixed-batch ids: DIFFER"
  date +%T ;;
esac
