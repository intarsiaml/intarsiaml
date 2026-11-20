#!/bin/zsh
# copyflipm served gate, inside a fleetq slot on endor: `serve --max-batch 8 --spec mtp` (and `--max-batch 1`) with INTARSIA_SPEC_COPY unset
# starts (as base does) and answers one request; each server is stopped by its PID.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-copyflipm/raw/serve; mkdir -p $RAW
BASE=$W/target-cf-base/intarsia-base; NEW=$W/target-cf/release/intarsia; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; PORT=8397; fails=0
try() { local n=$1 b=$2 mb=$3; shift 3
  env "$@" $b serve --backend metal --model $M --model-id m --max-batch $mb --max-seq 4096 --spec mtp --spec-k 2 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $RAW/$n.log 2>&1 < /dev/null &
  local pid=$! up=0
  for i in $(seq 1 100); do kill -0 $pid 2>/dev/null || break; curl -sf -m 3 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { up=1; break; }; sleep 3; done
  if [ $up = 1 ]; then curl -s -m 120 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' -d '{"model":"m","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":24,"temperature":0}' > $RAW/$n.reply.json
    echo "$n: up, reply: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; print(d["usage"]["completion_tokens"], "tokens:", (m["content"] or m.get("reasoning_content",""))[:50].replace("\n"," "))' $RAW/$n.reply.json 2>&1)"
    echo "$n: log copy lines: $(grep -ac 'copy drafts on' $RAW/$n.log)"; kill $pid; wait $pid 2>/dev/null
  else echo "$n: did not come up: $(grep -a -m1 -iE 'error|unsupported|needs' $RAW/$n.log | cut -c1-200)"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; fi
  echo "$n: up=$up"; echo $up > $RAW/$n.up; }
try base-mb8 $BASE 8 X=1; try new-mb8 $NEW 8 X=1
try new-mb8-bm $NEW 8 INTARSIA_BATCHED_MTP=4
try new-mb1 $NEW 1 X=1
cmp -s $RAW/base-mb8.up $RAW/new-mb8.up && echo "PASS serve mb8: new up=$(cat $RAW/new-mb8.up) as base" || { echo "FAIL serve mb8: up differs from base"; fails=$((fails+1)); }
grep -aq 'copy drafts on' $RAW/new-mb8.log $RAW/new-mb8-bm.log && { echo "FAIL serve mb8: copy drafts on at mb8"; fails=$((fails+1)); } || echo "PASS serve mb8: no copy drafts at mb8"
[ "$(cat $RAW/new-mb1.up)" = 1 ] && grep -aq 'copy drafts on.*default on Metal' $RAW/new-mb1.log && echo "PASS serve mb1: up, copy drafts on by default" || { echo "FAIL serve mb1"; fails=$((fails+1)); }
# the served plan: every startup line of the new mb8 server equals the base's (timings, ports and paths blanked)
nl() { sed -E 's/0x[0-9a-f]+/0xN/g; s/[0-9]+\.[0-9]+ ?(s|ms|MB|GB|GiB)/T/g; s/[0-9]{9,}/N/g' $1 | grep -avE 'listening|ready|^$' | sort; }
nl $RAW/base-mb8.log > $RAW/base-mb8.norm; nl $RAW/new-mb8.log > $RAW/new-mb8.norm
cmp -s $RAW/base-mb8.norm $RAW/new-mb8.norm && echo "PASS serve mb8: startup lines equal base ($(wc -l < $RAW/new-mb8.norm | tr -d ' ') lines)" || { echo "FAIL serve mb8: startup lines differ from base"; diff $RAW/base-mb8.norm $RAW/new-mb8.norm | head -10; fails=$((fails+1)); }
echo "serve fails=$fails"
