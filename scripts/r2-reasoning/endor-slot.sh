#!/bin/bash
# R2 reasoning parser on endor (Metal, Qwen3-8B BF16), base vs head.
#   fleetq run --host endor --mem 8 --label "R2 build" -- bash scripts/r2-reasoning/endor-slot.sh build OUT BASE HEAD
#   fleetq run --host endor --exclusive --mem 24 --label "R2 e2e" -- bash scripts/r2-reasoning/endor-slot.sh run OUT BASE HEAD
# BASE and HEAD are git revs of this repo. `run`, per arm (a fresh serve each):
#   greedy ids of 4 requests (thinking off, default, thinking on, thinking-off constrained), base vs head must match;
#   head only: e2e.py (50 thinking-on json_schema requests + 5 Anthropic tool_use requests) at concurrency 4.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
MODE=$1 O=$2 BASE=$3 HEAD=$4
M=/Users/josef/models/Qwen3-8B
PORT=8108 NAME=qwen3-8b SPID=
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

serve() { # arm
  "$O/target-$1/release/intarsia" serve --model $M --backend metal --max-seq 8192 --profile-every 0 --no-supervisor \
    --no-ui --host 127.0.0.1 --port $PORT --served-model-name $NAME > "$O/serve-$1.log" 2>&1 < /dev/null &
  SPID=$!
  for _ in $(seq 1 240); do
    curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && { echo "serve $1 up $(date +%T)"; return 0; }
    kill -0 $SPID 2>/dev/null || break
    sleep 1
  done
  tail -20 "$O/serve-$1.log"; return 1
}

unserve() { [ -n "$SPID" ] && { kill $SPID; wait $SPID 2>/dev/null; }; SPID=; sleep 2; }

# greedy ids (and content) of one request; $3 = JSON fragment spliced into the body
ids() { # arm tag fragment
  curl -sf http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' -d '{"messages": [{"role": "user", "content": "Write a short story about a lighthouse keeper robot."}], "max_tokens": 200, "temperature": 0, "return_token_ids": true '"$3"'}' \
    < /dev/null | python3 -I -c 'import json,sys; c=json.load(sys.stdin)["choices"][0]; print(c["token_ids"])' > "$O/ids-$1-$2.txt"
  echo "ids $1 $2: $(md5 -q "$O/ids-$1-$2.txt" | cut -c1-8) n=$(tr ',' '\n' < "$O/ids-$1-$2.txt" | wc -l)"
}
cids() { # arm: greedy constrained, thinking off
  curl -sf http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' -d '{"messages": [{"role": "user", "content": "Make up a fictional software engineer."}], "max_tokens": 200, "temperature": 0, "return_token_ids": true, "chat_template_kwargs": {"enable_thinking": false}, "response_format": {"type": "json_schema", "json_schema": {"name": "p", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"], "additionalProperties": false}, "strict": true}}}' \
    < /dev/null | python3 -I -c 'import json,sys; c=json.load(sys.stdin)["choices"][0]; print(c["token_ids"], c["message"]["content"])' > "$O/ids-$1-constrained-off.txt"
  echo "ids $1 constrained-off: $(md5 -q "$O/ids-$1-constrained-off.txt" | cut -c1-8)"
}
all_ids() {
  ids $1 off ', "chat_template_kwargs": {"enable_thinking": false}'
  ids $1 default ''
  ids $1 on ', "chat_template_kwargs": {"enable_thinking": true}'
  cids $1
}

case "$MODE" in
  build) build ;;
  anth) # head only: the Anthropic cases of e2e.py
    build; serve head || exit 3
    uv run --script "$R/scripts/r2-reasoning/e2e.py" --url http://127.0.0.1:$PORT --out "$O/e2e-anthropic.json" --only anthropic --concurrency 2 > "$O/e2e-anthropic.log" 2>&1
    echo "e2e anthropic rc=$? $(tail -1 "$O/e2e-anthropic.log")"; unserve ;;
  run)
    date +%T
    build
    for arm in base head; do
      serve $arm || exit 3
      all_ids $arm
      if [ $arm = head ]; then
        curl -sf http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' -d '{"messages": [{"role": "user", "content": "What is 17*23?"}], "max_tokens": 1500, "temperature": 0, "chat_template_kwargs": {"enable_thinking": true}}' < /dev/null > "$O/sample-thinking.json"
        uv run --script "$R/scripts/r2-reasoning/e2e.py" --url http://127.0.0.1:$PORT --out "$O/e2e-c4.json" --concurrency 4 > "$O/e2e-c4.log" 2>&1
        echo "e2e rc=$? $(tail -1 "$O/e2e-c4.log")"
      fi
      unserve
    done
    for t in off default on constrained-off; do
      cmp -s "$O/ids-base-$t.txt" "$O/ids-head-$t.txt" && echo "ids $t: IDENTICAL" || echo "ids $t: DIFFER"
    done
    date +%T ;;
esac
