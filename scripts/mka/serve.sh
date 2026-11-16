#!/bin/zsh
# Lane mka served runs. Under fleetq --quiet gpu:
#   serve.sh ident                      gate 4 served: 35B --max-batch 8, 4 and 8 streams == solo, arms base / off / on
#   serve.sh speed <35|27> <spec> <sizes> [reps] [order]   ABBA localeval speed, fresh server per arm (order default "off on on off")
set -u
POLICY="$(cat /opt/code/intarsiaml/.worktrees/mka/scripts/mka/policy.env 2>/dev/null)"
W=/opt/code/intarsiaml/.worktrees/mka; R=$W/docs/results/2026-11-15-mka-metal-kv-once/raw
BASE=$W/target/base/release/intarsia; BIN=$W/target/release/intarsia
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
PORT=18631
TP=""
start() { # outdir arm model extra...
  local O=$1 arm=$2 model=$3; shift 3
  local B=$BIN E=(X=1); [ $arm = base ] && B=$BASE; [ $arm = on ] && E=(INTARSIA_METAL_ATTN_KV_ONCE=1 ${=POLICY})
  mkdir -p $O
  CMD=($B serve --backend metal --model $model --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@")
  env $E $CMD > $O/server.out 2> $O/server.err < /dev/null &
  TP=$!
  local up=0
  for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $O"; tail -5 $O/server.err; kill $TP 2>/dev/null; return 1; }
}
stop() { kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; }
trap 'stop' EXIT INT TERM

[ -f $R/gates/SMOKE_OK ] || { echo "no SMOKE_OK marker: skipped"; exit 3; }
echo "policy of the on arm: [$POLICY]"
case ${1:?cmd} in
  ident)
    O=$R/gates/served; mkdir -p $O
    for arm in base off on; do
      start $O/$arm $arm $M35 --max-batch 8 || continue
      python3 $W/scripts/mka/served_ident.py run --url http://127.0.0.1:$PORT/v1 --text $PR/prompt-16384.txt --out $O/$arm.json 2>&1 | sed "s/^/$arm: /"
      stop
    done
    python3 $W/scripts/mka/served_ident.py compare $O/base.json $O/off.json
    python3 $W/scripts/mka/served_ident.py compare $O/off.json $O/on.json
    ;;
  speed)
    m=$2; spec=$3; sizes=$4; reps=${5:-2}; order=${6:-"off on on off"}
    model=$M35; [ $m = 27 ] && model=$M27
    SA=(); [ $spec = off ] && SA=(--spec off)
    i=0
    for arm in ${=order}; do
      i=$((i+1)); O=$R/speed/$m-$spec-$sizes/$i-$arm
      start $O $arm $model $SA || continue
      (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes $sizes --gen 256 --reps $reps \
         --thinking off --no-unim --label "mka-$m-$spec-$arm" --recipe "${CMD[*]} kvo=$arm" > $O/localeval.out 2> $O/localeval.err; echo "localeval exit $?")
      echo "## $m spec=$spec arm $i $arm: $(grep -a 'spec:' $O/server.err | tail -1 | grep -oE 'passes=.*')"
      grep -aE '^\| *[0-9]+ \|' $O/localeval.out
      stop
    done
    ;;
esac
echo "## done $*"
