#!/bin/bash
# MDEC window 4: gates + final A/B. usage: w4.sh OUT MAINBIN
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mdec; O=$1; MAIN=$2; mkdir -p $O; BIN=$W/.lane/target/release/intarsia
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
cd $W; export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$W/.lane/target
echo "## gates $(date +%T)"
INTARSIA_REQUIRE_GPU=1 cargo test --release --locked -p intarsia-backend-metal --test rows_exact -- --test-threads=1 > $O/rows_exact.log 2>&1; echo "rows_exact exit $?"; grep '^test result' $O/rows_exact.log
INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 cargo test --release --locked -p intarsia-models --test batch_metal_real -- --test-threads=1 > $O/gc.log 2>&1; echo "batch_metal_real exit $?"; grep -E '^test |^test result' $O/gc.log
INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 cargo test --release --locked -p intarsia-models --test batch_metal > $O/dense.log 2>&1; echo "batch_metal (8B) exit $?"; grep '^test result' $O/dense.log
echo "## m1 pf A/B $(date +%T)"
P=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
gen() { local tag=$1 b=$2; shift 2; env X=1 "$@" $b generate --model $M35 --backend metal --max-seq 20000 --profile-every 0 --temperature 0 --max-tokens 257 --ignore-eos --prompt "$(cat $P)" > $O/$tag.out 2> $O/$tag.err
  echo "$tag $(grep -o 'decode [0-9]* ms ([0-9.]* tok/s)' $O/$tag.err | tail -1) md5 $(md5 -q $O/$tag.out)"; }
for r in 1 2 3; do gen p$r-main $MAIN; gen p$r-off $BIN; gen p$r-pf $BIN INTARSIA_METAL_FP8_M1_PF=1; done
PORT=18474
srv() { local tag=$1 mb=$2; shift 2
  local cmd="$BIN serve --backend metal --model $M35 --max-batch $mb --served-model-name intarsia --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT"
  echo "## $tag $(date +%T)"; env X=1 $cmd > $O/$tag-server.out 2> $O/$tag-server.err & local sp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 1024 --gen 512 --reps 1 \
     --concurrency 1,4,8,16 --thinking off --label "mdec-w4-$tag" --recipe "$cmd" > $O/$tag-localeval.out 2> $O/$tag-localeval.err; echo "localeval exit $?")
  grep -E '^\| (1|4|8|16) \|' $O/$tag-localeval.out
  kill $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}
srv r1-mb8 8; srv r1-mb16 16; srv r2-mb16 16; srv r2-mb8 8; srv r3-mb8 8; srv r3-mb16 16
echo "## done $(date +%T)"
