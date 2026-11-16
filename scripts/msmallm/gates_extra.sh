#!/bin/zsh
# Lane msmallm gates, second pass (gates.sh's TAIL prompt tokenised to 2474 = 2048 + 426: outside the window, so its ids and route
# steps did not exercise the routing). Same arms and helpers as gates.sh; prompts sized to land in the window:
#   TAIL2 about 8000 characters = 2048 + about 80 rows; INWIN (125 tokens) for the routing trace.
#   served1: the default engine (MTP k2, max-batch 1: `--max-batch` > 1 on Metal serves without MTP), 1 / 4 / 9 / 16 concurrent
#            requests (queued), ids main / unset / zero and the served plan.
# usage (under fleetq): gates_extra.sh route | ids | served1
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/msmallm; R=$W/docs/results/2026-11-16-msmallm/raw/gates; mkdir -p $R
BASE=$W/target/base/release/intarsia; BIN=$W/target/release/intarsia
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
INWIN="$(head -c 520 $PR/prompt-16384.txt)"
TAIL2="$(tail -c +40001 $PR/prompt-16384.txt | head -c 8000)"
arm_env() { case $1 in base) B=$BASE; E=(X=1) ;; unset) B=$BIN; E=(X=1) ;; zero) B=$BIN; E=(INTARSIA_METAL_MXFP4_NAXD=0) ;; esac; }
ids() { # pn prompt spec arms...
  local pn=$1 ptxt=$2 spec=$3; shift 3
  local SA=(); [ $spec = off ] && SA=(--spec off)
  local ref="" ok=1 line="" pt=""
  for arm in "$@"; do
    arm_env $arm
    local o=$R/ids-$pn-$spec-$arm
    env $E $B generate --model $M27 --backend metal --max-seq 40000 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids $SA \
      --prompt "$ptxt" > $o.out 2> $o.err
    local got="$(grep -a '^ids' $o.err | md5 -q)" n=$(grep -a '^ids' $o.err | grep -oE '[0-9]+' | wc -l | tr -d ' ')
    pt="$(grep -aoE '^prompt: [0-9]+ tokens' $o.err)"
    [ -z "$ref" ] && ref=$got; [ "$got" = "$ref" ] && [ $n -ge 200 ] || ok=0
    line="$line | $arm $got[1,8] n=$n $(grep -aoE 'prefill [0-9]+ ms' $o.err | tail -1)"
  done
  echo "$( [ $ok = 1 ] && echo SAME || echo DIFFER ) 27b $pn spec=$spec ($pt)$line"
}
PORT=18674
start() { local O=$1 arm=$2; shift 2; arm_env $arm; mkdir -p $O
  env $E $B serve --backend metal --model $M27 --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT "$@" > $O/server.out 2> $O/server.err < /dev/null &
  TP=$!
  for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0; kill -0 $TP 2>/dev/null || break; sleep 1; done
  echo "SERVER FAILED $O"; kill $TP 2>/dev/null; return 1
}
stop() { kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; }
case ${1:?step} in
  served1)
    for arm in base unset zero; do
      start $R/served-mtp1-$arm $arm || continue
      curl -s http://127.0.0.1:$PORT/v1/intarsia/plan > $R/plan-served-mtp1-$arm.json
      python3 $W/scripts/msmallm/served_ids.py run --url http://127.0.0.1:$PORT/v1 --text $PR/prompt-16384.txt --out $R/served-mtp1-$arm.json 2>&1 | sed "s/^/$arm: /"
      stop
      echo "$arm: $(grep -a 'specul' $R/served-mtp1-$arm/server.err | head -1 | cut -c1-80)"
    done
    python3 $W/scripts/msmallm/served_ids.py compare $R/served-mtp1-base.json $R/served-mtp1-unset.json
    python3 $W/scripts/msmallm/served_ids.py compare $R/served-mtp1-unset.json $R/served-mtp1-zero.json
    for a in unset zero; do echo "plan served mtp1 base vs $a: $(diff <(python3 -m json.tool $R/plan-served-mtp1-base.json) <(python3 -m json.tool $R/plan-served-mtp1-$a.json) | grep -c '^[<>]') differing lines"; done
    ;;
  route)
    for arm in unset zero; do arm_env $arm
      for pn in inwin tail2; do
        [ $pn = inwin ] && P="$INWIN" || P="$TAIL2"
        env $E INTARSIA_METAL_PROFILE=cb:$R/route-$pn-$arm.jsonl INTARSIA_METAL_COMMIT_EVERY=1 $B generate --model $M27 --backend metal --max-seq 40000 \
          --profile-every 0 --temperature 0 --max-tokens 2 --ignore-eos --spec off --prompt "$P" > $R/route-$pn-$arm.out 2> $R/route-$pn-$arm.err
        echo "route $pn $arm: $(python3 -c "
import json,collections
c=collections.Counter()
for l in open('$R/route-$pn-$arm.jsonl'):
    if l.startswith('{'):
        r=json.loads(l)
        if r['op'].startswith('matmul_mxfp4_nax'): c[(r['op'],r['t'])]+=1
print(sorted(c.items()))")"
        gzip -f $R/route-$pn-$arm.jsonl
      done
    done
    ;;
  ids)
    for spec in default off; do ids tail2 "$TAIL2" $spec base unset zero; done
    ;;
esac
echo "## done $*"
