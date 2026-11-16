#!/bin/zsh
# Lane msmallm gates (not timing runs). Run under fleetq, one step per job:
#   gates.sh route | ids | served <off|mtp> | plan
# Arms: base = origin/main binary (target/base), unset = this branch with INTARSIA_METAL_MXFP4_NAXD unset (small-M routing on),
# zero = this branch with INTARSIA_METAL_MXFP4_NAXD=0 (the scratch path at every M).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/msmallm; R=$W/docs/results/2026-11-16-msmallm/raw/gates; mkdir -p $R
BASE=$W/target/base/release/intarsia; BIN=$W/target/release/intarsia
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
SHORT="The history of the printing press begins"
INWIN="$(head -c 520 $PR/prompt-16384.txt)"                 # about 120 tokens: every NAX call in the window
TAIL="$(tail -c +40001 $PR/prompt-16384.txt | head -c 9300)"  # about 2150 tokens: a 2048 chunk + an in-window remainder
P4K="$(cat $PR/prompt-4096.txt)"
PORT=18672

arm_env() { case $1 in base) B=$BASE; E=(X=1) ;; unset) B=$BIN; E=(X=1) ;; zero) B=$BIN; E=(INTARSIA_METAL_MXFP4_NAXD=0) ;; esac; }

ids() { # pn prompt spec arms...
  local pn=$1 ptxt=$2 spec=$3; shift 3
  local SA=(); [ $spec = off ] && SA=(--spec off)
  local ref="" ok=1 line=""
  for arm in "$@"; do
    arm_env $arm
    local o=$R/ids-$pn-$spec-$arm
    env $E $B generate --model $M27 --backend metal --max-seq 40000 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids $SA \
      --prompt "$ptxt" > $o.out 2> $o.err
    local got="$(grep -a '^ids' $o.err | md5 -q)" n=$(grep -a '^ids' $o.err | grep -oE '[0-9]+' | wc -l | tr -d ' ')
    local pf="$(grep -aoE 'prompt [0-9]+ tok' $o.err | head -1)" sp="$(grep -a 'spec:' $o.err | tail -1 | grep -oE 'accepted=[0-9]+')"
    [ -z "$ref" ] && ref=$got; [ "$got" = "$ref" ] && [ $n -ge 200 ] || ok=0
    line="$line | $arm $got[1,8] n=$n $sp"
  done
  echo "$( [ $ok = 1 ] && echo SAME || echo DIFFER ) 27b $pn spec=$spec ($pf)$line"
}

start() { # outdir arm extra...
  local O=$1 arm=$2; shift 2; arm_env $arm; mkdir -p $O
  env $E $B serve --backend metal --model $M27 --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT "$@" > $O/server.out 2> $O/server.err < /dev/null &
  TP=$!
  for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0; kill -0 $TP 2>/dev/null || break; sleep 1; done
  echo "SERVER FAILED $O: $(grep -a -i error $O/server.err | tail -2)"; kill $TP 2>/dev/null; return 1
}
stop() { kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; }

case ${1:?step} in
  route)
    # the routing fires: with the knob unset an in-window call has two dispatches (x prep, GEMM), no `matmul_mxfp4_nax#2`
    for arm in unset zero; do arm_env $arm
      env $E INTARSIA_METAL_PROFILE=cb:$R/route-$arm.jsonl INTARSIA_METAL_COMMIT_EVERY=1 $B generate --model $M27 --backend metal --max-seq 40000 \
        --profile-every 0 --temperature 0 --max-tokens 2 --ignore-eos --spec off --prompt "$TAIL" > $R/route-$arm.out 2> $R/route-$arm.err
      echo "route $arm: $(python3 -c "
import json,collections
c=collections.Counter()
for l in open('$R/route-$arm.jsonl'):
    if l.startswith('{'):
        r=json.loads(l)
        if r['op'].startswith('matmul_mxfp4_nax'): c[(r['op'],r['t'])]+=1
print(sorted(c.items()))")"
      gzip -f $R/route-$arm.jsonl
    done
    ;;
  ids)
    for spec in default off; do
      ids short "$SHORT" $spec base unset zero
      ids inwin "$INWIN" $spec base unset zero
      ids tail "$TAIL" $spec base unset zero
      ids 4k "$P4K" $spec base unset zero
    done
    ;;
  served)
    spec=$2; SA=(--max-batch 8); [ $spec = off ] && SA=(--max-batch 8 --spec off)
    for arm in base unset zero; do
      start $R/served-$spec-$arm $arm $SA || { [ $spec = mtp ] && SA=() && start $R/served-$spec-$arm $arm || continue; }
      echo "served $spec $arm: flags [$SA]"
      curl -s http://127.0.0.1:$PORT/v1/intarsia/plan > $R/plan-served-$spec-$arm.json
      python3 $W/scripts/msmallm/served_ids.py run --url http://127.0.0.1:$PORT/v1 --text $PR/prompt-16384.txt --out $R/served-$spec-$arm.json 2>&1 | sed "s/^/$arm: /"
      stop
    done
    python3 $W/scripts/msmallm/served_ids.py compare $R/served-$spec-base.json $R/served-$spec-unset.json
    python3 $W/scripts/msmallm/served_ids.py compare $R/served-$spec-unset.json $R/served-$spec-zero.json
    for a in unset zero; do echo "plan served $spec base vs $a: $(diff <(python3 -m json.tool $R/plan-served-$spec-base.json) <(python3 -m json.tool $R/plan-served-$spec-$a.json) | grep -c '^[<>]') differing lines"; done
    ;;
esac
echo "## done $*"
