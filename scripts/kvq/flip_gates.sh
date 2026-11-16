#!/bin/zsh
# Lane kvq: gates of the default flip. Arms: z = INTARSIA_METAL_ATTN_KV_ONCE=0 (old launches), d = knob unset (new default), same binary.
#   flip_gates.sh ids | plan     (run each under fleetq --quiet gpu,cpu)
set -u
W=/opt/code/intarsiaml/.worktrees/kvq; R=$W/docs/results/2026-11-16-kvq/raw/flip; mkdir -p $R
BIN=${BIN:-$W/target/kvq-flip}
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
SHORT="The history of the printing press begins"
arm_env() { case $1 in z) E=(INTARSIA_METAL_ATTN_KV_ONCE=0) ;; d) E=(X=1) ;; esac }
ids() { # tag model prompt-name prompt-text spec
  local tag=$1 model=$2 pn=$3 ptxt=$4 spec=$5 SA=() ref="" ok=1 line=""
  [ $spec = off ] && SA=(--spec off)
  for arm in z d; do
    arm_env $arm; local o=$R/ids-$tag-$pn-$spec-$arm
    env $E $BIN generate --model $model --backend metal --max-seq 40000 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids $SA \
      --prompt "$ptxt" > $o.out 2> $o.err
    local got="$(grep -a '^ids' $o.err | md5 -q)" n=$(grep -a '^ids' $o.err | grep -oE '[0-9]+' | wc -l | tr -d ' ')
    local sp="$(grep -a 'spec:' $o.err | tail -1 | grep -oE 'passes=[0-9]+ proposed=[0-9]+ accepted=[0-9]+')"
    [ -z "$ref" ] && ref=$got; [ "$got" = "$ref" ] && [ $n -ge 200 ] || ok=0
    line="$line | $arm ${got[1,8]} n=$n $sp"
  done
  echo "$( [ $ok = 1 ] && echo SAME || echo DIFFER ) $tag $pn spec=$spec$line"
}
case ${1:?step} in
  ids)
    for spec in default off; do
      ids 35b $M35 short "$SHORT" $spec; ids 35b $M35 4k "$(cat $PR/prompt-4096.txt)" $spec; ids 35b $M35 32k "$(cat $PR/prompt-32768.txt)" $spec
      ids 27b $M27 short "$SHORT" $spec; ids 27b $M27 4k "$(cat $PR/prompt-4096.txt)" $spec; ids 27b $M27 32k "$(cat $PR/prompt-32768.txt)" $spec
    done ;;
  plan)
    for m in $M35 $M27; do for spec in default off; do
      SA=(); [ $spec = off ] && SA=(--spec off)
      for arm in z d; do
        arm_env $arm; o=$R/plan-${m:t}-$spec-$arm
        env $E $BIN serve --backend metal --model $m --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18651 $SA > $o.srv 2>&1 < /dev/null &
        P=$!; for i in {1..300}; do curl -sf -m 2 http://127.0.0.1:18651/health >/dev/null 2>&1 && break; sleep 1; done
        curl -s http://127.0.0.1:18651/v1/intarsia/plan | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin),indent=1,sort_keys=True))' > $o.json
        kill -INT $P; wait $P 2>/dev/null; sleep 3
      done
      echo "plan ${m:t} $spec: $(diff $R/plan-${m:t}-$spec-z.json $R/plan-${m:t}-$spec-d.json | grep -c '^[<>]') differing lines"
      diff $R/plan-${m:t}-$spec-z.json $R/plan-${m:t}-$spec-d.json | head -20
    done; done ;;
esac
echo "## done $*"
