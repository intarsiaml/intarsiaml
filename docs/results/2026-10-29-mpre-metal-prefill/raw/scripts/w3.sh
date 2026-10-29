#!/bin/bash
# MPRE window 3: localeval interleaved A/B (35B: base / c1 / c1 + chunk 4096; 8B: base / c1), per-knob plain-prefill
# splits with a warm cb trace, then cargo test --workspace --release. Inside ONE fleetq slot. usage: w3.sh R35 R8
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
R35=${1:-3}; R8=${2:-2}
L=/opt/code/intarsiaml/.worktrees/mpre/.lane; C=$L/intarsia-c1; B=$L/intarsia-base; OUT=$L/w3; mkdir -p $OUT
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M8=$HOME/Models/Qwen3-8B; PORT=18437
/Users/josef/bin/macmon pipe -i 500 > $OUT/macmon.jsonl 2> /dev/null &
MM=$!
T0=$SECONDS
log() { echo "$(date +%T) +$(( (SECONDS - T0) / 60 ))m $*" | tee -a $OUT/marks.txt; }
log "start rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)"
md5 -q $B $C > $OUT/versions.txt

arm() { # tag model bin extra-serve-args
  local tag=$1 model=$2 bin=$3 extra=$4
  local cmd="$bin serve --backend metal --model $model --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT $extra"
  $cmd > $OUT/$tag-server.out 2> $OUT/$tag-server.err &
  local sp=$!
  (while true; do echo "$(date +%T) $(ps -axo pid,%cpu,comm -r | sed -n 2,4p | tr -s ' ' | tr '\n' ';') rustc=$(pgrep -x rustc | wc -l)"; sleep 4; done) > $OUT/$tag-load.log 2>&1 &
  local lp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 1024,4096,16384 --gen 256 --reps 3 \
     --thinking off --label "mpre-$tag" --recipe "$cmd" > $OUT/$tag-localeval.out 2> $OUT/$tag-localeval.err)
  log "$tag localeval exit $?: $(grep -E '^\| (1[0-9]{3}|4[0-9]{3}|16[0-9]{3}) ' $OUT/$tag-localeval.out | cut -d'|' -f2,4,6 | tr -s ' ' | tr '\n' ';')"
  kill $lp 2>/dev/null; kill $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}

# 35B: rotate the arm order each round
for r in $(seq 1 $R35); do
  case $((r % 3)) in 1) order="base c1 c1k4";; 2) order="c1k4 base c1";; 0) order="c1 c1k4 base";; esac
  for a in $order; do
    case $a in base) arm $r-35-base $M35 $B "";; c1) arm $r-35-c1 $M35 $C "";; c1k4) arm $r-35-c1k4 $M35 $C "--prefill-chunk 4096";; esac
  done
done
# 8B
for r in $(seq 1 $R8); do
  if [ $((r % 2)) = 1 ]; then order="base c1"; else order="c1 base"; fi
  for a in $order; do
    case $a in base) arm $r-8-base $M8 $B "";; c1) arm $r-8-c1 $M8 $C "";; esac
  done
done

# per-knob splits, warm (generate --repeat 3), 4k and 16k; cb trace of c1 at 4k
P=/opt/code/intarsiaml/.worktrees/mpre/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for ctx in 4096 16384; do
  for a in off d256 nv both; do
    case $a in off) E="INTARSIA_METAL_FLASH_NAX_D256=0 INTARSIA_METAL_NVFP4_NAX=0";; d256) E="INTARSIA_METAL_NVFP4_NAX=0";; nv) E="INTARSIA_METAL_FLASH_NAX_D256=0";; both) E="A=1";; esac
    env $E $C generate --model $M35 --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 --ignore-eos --profile-every 0 \
      --repeat 3 --prompt "$(cat $P/prompt-$ctx.txt)" > /dev/null 2> $OUT/split-$a-$ctx.err
    log "split $a $ctx: $(grep -o 'prefill [0-9]* ms' $OUT/split-$a-$ctx.err | tr '\n' ' ')"
  done
done
env INTARSIA_METAL_PROFILE=cb:$OUT/cb-c1-4k.jsonl $C generate --model $M35 --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 \
  --ignore-eos --profile-every 0 --repeat 2 --prompt "$(cat $P/prompt-4096.txt)" > /dev/null 2> $OUT/cb-c1-4k.err
env INTARSIA_METAL_PROFILE=kernel:$OUT/kernel-c1-16k.jsonl $C generate --model $M35 --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 \
  --ignore-eos --profile-every 0 --prompt "$(cat $P/prompt-16384.txt)" > /dev/null 2> $OUT/kernel-c1-16k.err
env INTARSIA_METAL_PROFILE=kernel:$OUT/kernel-c1-4k.jsonl $C generate --model $M35 --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 \
  --ignore-eos --profile-every 0 --prompt "$(cat $P/prompt-4096.txt)" > /dev/null 2> $OUT/kernel-c1-4k.err
log "traces done"
if [ $((SECONDS - T0)) -lt 1900 ]; then
  (cd /opt/code/intarsiaml/.worktrees/mpre && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$L/target \
    cargo test --workspace --release --no-fail-fast > $OUT/ws-test.log 2>&1); log "workspace tests exit $?: $(grep -E '^test result' $OUT/ws-test.log | awk '{p+=$4; f+=$6; i+=$8} END {print p" passed, "f" failed, "i" ignored"}')"
else log "SKIP workspace tests (time)"; fi
kill $MM; gzip -f $OUT/*.jsonl
log "end rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)"
