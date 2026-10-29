#!/bin/bash
# chunk4k follow-ups: (2) long-prompt engine test in its default invocation, (1) busy_chunk=2048: plan diff, ids, busy-chunk log line, single-stream 4k prefill.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/chunk4k; cd $W; O=$W/.lane/w5; mkdir -p $O
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$W/.lane/target INTARSIA_REPO=$W
B=$W/.lane/intarsia-busy; M=$HOME/Models/Qwen3.6-35B-A3B-NVFP4; IP=$W/scripts/mpf/ids-prompts; PORT=18493
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $O/job.log; }
INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M \
  cargo test --release --locked -p intarsia-models --test batch_metal_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --exact --nocapture > $O/long-default.log 2>&1
log "long-prompt test (default invocation) exit $?: $(grep -a -E '^test result' $O/long-default.log)"
grep -a -E "LONGENG" $O/long-default.log | cut -c1-200 | tee -a $O/job.log
# plan show vs the 4096-only plan of the first PR commit (same flags)
$B plan show --backend metal --model $M --mem-budget 40 --max-seq 20480 --json </dev/null > $O/plan-busy-20480.json 2> $O/plan-busy.err
diff .lane/plan/new-20480.json $O/plan-busy-20480.json > $O/plan-busy.diff; log "plan show diff vs busy=4096: $(wc -l < $O/plan-busy.diff) lines"; cat $O/plan-busy.diff | tee -a $O/job.log
# ids: short-0 and 4k-0 with the recipe (generate is single-sequence: busy_chunk is not on its path), against the saved 2048 ids
for n in short-0 4k-0; do
  case $n in short*) MS=4096;; *) MS=8192;; esac
  $B generate -m $M --backend metal --max-seq $MS --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids -p "$(cat $IP/$n.txt)" --max-tokens 256 --spec off > $O/ids-$n.out 2> $O/ids-$n.err </dev/null
  grep -a '^ids' $O/ids-$n.err | sed -E 's/^ids(\[[0-9]+\])?: //' | sort -u > $O/ids-$n.uniq
  grep -a '^ids' .lane/w1/ids-$n-c2048.err | sed -E 's/^ids(\[[0-9]+\])?: //' | sort -u > $O/ids-$n.ref2048
  cmp -s $O/ids-$n.uniq $O/ids-$n.ref2048 && log "ids $n: identical to the 2048 run ($(grep -a 'prefill chunk' $O/ids-$n.err | head -1))" || log "ids $n: DIFFERS"
done
# serve: the busy chunk the engine reports at --max-batch 4, and single-stream prefill 1k/4k, recipe default vs --prefill-chunk 2048
srv() { local tag=$1 mb=$2 extra=$3 bench=${4:-}
  $B serve --backend metal --model $M $mb --served-model-name intarsia --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT $extra > $O/$tag-server.out 2> $O/$tag-server.err & local sp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  log "$tag: $(grep -a -E 'prefill chunk|busy prefill chunk' $O/$tag-server.err | tr '\n' ';')"
  if [ -n "$bench" ]; then (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 1024,4096 --gen 128 --reps 3 --thinking off --label "chunk4k-$tag" --recipe "$B serve ... $mb $extra" > $O/$tag-localeval.out 2> $O/$tag-localeval.err); log "$tag localeval exit $?"; fi
  kill $sp; wait $sp 2>/dev/null; for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done; }
srv mb4 "--max-batch 4" "" ""
srv single-default "" "" bench
srv single-c2048 "" "--prefill-chunk 2048" bench
srv single-default2 "" "" bench
log done
