#!/bin/bash
# MDEC window 2. usage: w2.sh OUT
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mdec; O=$1; mkdir -p $O; BIN=$W/.lane/target/release/intarsia
D=$W/.lane/target/release/deps
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
echo "## rows_exact $(date +%T)"
INTARSIA_REQUIRE_GPU=1 $(ls -t $D/rows_exact-* | grep -v '\.d$' | head -1) --test-threads=1 > $O/rows_exact.log 2>&1; echo "rows_exact exit $?"; grep -E '^test result|FAILED|panicked' $O/rows_exact.log | head
echo "## step timing $(date +%T)"
INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 $(ls -t $D/batch_metal_real-* | grep -v '\.d$' | head -1) --ignored real_35b_batch_step_timing --nocapture > $O/step.log 2>&1; echo "step exit $?"; grep TIMING $O/step.log
echo "## fp8 narrow A/B $(date +%T)"
P=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
gen() { local tag=$1; shift; env X=1 "$@" $BIN generate --model $M35 --backend metal --max-seq 20000 --profile-every 0 --temperature 0 --max-tokens 257 --ignore-eos --prompt "$(cat $P)" > $O/$tag.out 2> $O/$tag.err
  echo "$tag $(grep -o 'decode [0-9]* ms ([0-9.]* tok/s)' $O/$tag.err | tail -1) md5 $(md5 -q $O/$tag.out)"; }
for r in 1 2 3; do
  for v in off r2u4 r1u8 r2u8; do gen n$r-$v INTARSIA_METAL_FP8_M1_NARROW=$v; done
done
for v in off r2u4 r1u8; do gen ce1-$v INTARSIA_METAL_FP8_M1_NARROW=$v INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/ce1-$v.jsonl; done
PORT=18472
srv() { local tag=$1 mb=$2 conc=$3; shift 3
  local cmd="$BIN serve --backend metal --model $M35 --max-batch $mb --served-model-name intarsia --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT"
  echo "## $tag $(date +%T)"; env X=1 "$@" $cmd > $O/$tag-server.out 2> $O/$tag-server.err & local sp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 1024 --gen 512 --reps 1 \
     --concurrency $conc --thinking off --label "mdec-w2-$tag" --recipe "$* $cmd" > $O/$tag-localeval.out 2> $O/$tag-localeval.err; echo "localeval exit $?")
  grep -A6 '^| streams' $O/$tag-localeval.out
  kill $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}
srv mb8 8 1,4,8,16
srv mb16 16 1,4,8,16
srv mb16h 16 16 INTARSIA_HOST_PROFILE=1
grep 'decode-only' $O/mb16h-server.err | tail -4 | cut -c1-200
gzip -f $O/*.jsonl
echo "## done $(date +%T)"
