#!/bin/zsh
# mac-plan baseline: where time goes on endor (M5 Max) for the 35B MoE and the 27B dense, Metal.
# Run inside a fleetq slot:  fleetq run --host endor --expect 45 --mem 30 --quiet gpu --label "mac-plan: ..." -- scripts/mac-plan/measure.sh [arms...]
# Arms: roof s35 s35off b35 p35 s27 s27off b27 p27   (default: all). Output under scripts/mac-plan/raw/<arm>/.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mac-plan; BIN=$W/target/release/intarsia; R=$W/scripts/mac-plan/raw
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
PORT=18611
ARMS=("$@"); [ $#ARMS = 0 ] && ARMS=(roof s35 s35off b35 p35 s27 s27off b27 p27)

serve_arm() { # tag model sizes gen reps concurrency -- extra serve args
  local tag=$1 model=$2 sizes=$3 gen=$4 reps=$5 conc=$6; shift 6
  local O=$R/$tag; mkdir -p $O
  local CMD=($BIN serve --backend metal --model $model --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@")
  echo "## $tag $(date +%T): ${CMD[*]}"
  $CMD > $O/server.out 2> $O/server.err < /dev/null &
  local TP=$! up=0
  for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $tag"; tail -5 $O/server.err; kill $TP 2>/dev/null; return 1; }
  local CA=(); [ "$conc" != 0 ] && CA=(--concurrency $conc)
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes $sizes --gen $gen --reps $reps $CA \
     --thinking off --no-unim --label "macplan-$tag" --recipe "${CMD[*]}" > $O/localeval.out 2> $O/localeval.err; echo "localeval exit $?")
  grep -a "spec:" $O/server.err | tail -2
  kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
  tail -25 $O/localeval.out
}

prof_arm() { # tag model ctx -- extra generate args : plain + cb-profile generate, 129 tokens
  local tag=$1 model=$2 ctx=$3; shift 3
  local O=$R/$tag; mkdir -p $O
  for mode in plain cb; do
    local E=(); [ $mode = cb ] && E=(INTARSIA_METAL_PROFILE=cb:$O/cb-$ctx.jsonl)
    env X=1 $E $BIN generate --model $model --backend metal --max-seq 40000 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos "$@" \
      --prompt "$(cat $PR/prompt-$ctx.txt)" > $O/$mode-$ctx.out 2> $O/$mode-$ctx.err
    echo "$tag $ctx $mode [$*] $(grep -aoE '(prefill|decode) [0-9]+ ms \([0-9.]+ tok/s\)' $O/$mode-$ctx.err | tr '\n' ' ')"
  done
}

for arm in $ARMS; do
  case $arm in
    roof)
      mkdir -p $R/roof
      (cd $W && cargo run --release --locked -q -p intarsia-backend-metal --example gemv_bw -- copy gemv > $R/roof/gemv_bw.out 2> $R/roof/gemv_bw.err; echo "gemv_bw exit $?"; tail -30 $R/roof/gemv_bw.out)
      (cd $W && NAX_M=256,2048 cargo run --release --locked -q -p intarsia-backend-metal --example gemm_nax -- bench > $R/roof/gemm_nax.out 2> $R/roof/gemm_nax.err; echo "gemm_nax exit $?"; grep RESULT $R/roof/gemm_nax.out | sort -t' ' -k2 | tail -30)
      ;;
    s35)    serve_arm s35 $M35 4096,32768 256 2 0 ;;
    s35off) serve_arm s35off $M35 4096,32768 256 2 0 --spec off ;;
    b35)    serve_arm b35 $M35 1024 256 1 2,4,8,16 --max-batch 16 ;;
    p35)    prof_arm p35 $M35 4096 --spec off; prof_arm p35 $M35 32768 --spec off; prof_arm p35mtp $M35 4096 ;;
    s27)    serve_arm s27 $M27 4096,32768 256 2 0 ;;
    s27off) serve_arm s27off $M27 4096,32768 256 1 0 --spec off ;;
    b27)    serve_arm b27 $M27 1024 128 1 2,4,8 --max-batch 8 ;;
    p27)    prof_arm p27 $M27 4096 --spec off; prof_arm p27mtp $M27 4096 ;;
  esac
done
echo "## done $(date +%T)"
