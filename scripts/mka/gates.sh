#!/bin/zsh
# Lane mka gates 2, 3, 4 (not timing runs). Run under fleetq --quiet gpu, one step per job:
#   gates.sh kernels | plan | ids35 | ids27 | ids8 | trace
# Arms: base = origin/main binary (target/base), off = this branch with the knob unset, on = INTARSIA_METAL_ATTN_KV_ONCE=1.
set -u
# extra environment of the `on` arm (the packing policy under test), one line of VAR=val words; empty = the shipped default
POLICY="$(cat /opt/code/intarsiaml/.worktrees/mka/scripts/mka/policy.env 2>/dev/null)"
W=/opt/code/intarsiaml/.worktrees/mka; R=$W/docs/results/2026-11-15-mka-metal-kv-once/raw/gates; mkdir -p $R
BASE=$W/target/base/release/intarsia; BIN=$W/target/release/intarsia
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; M8=$HOME/models/Qwen3-8B
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
SHORT="The history of the printing press begins"
# a second 4k prompt: the middle of the 16k prompt (as many characters as the 4k one)
N4=$(wc -c < $PR/prompt-4096.txt); P4B="$(tail -c +30001 $PR/prompt-16384.txt | head -c $N4)"

run_arm() { # arm -> sets E (env) and B (binary)
  case $1 in
    base) B=$BASE; E=(X=1) ;;
    off)  B=$BIN;  E=(X=1) ;;
    on)   B=$BIN;  E=(INTARSIA_METAL_ATTN_KV_ONCE=1 ${=POLICY}) ;;
  esac
}

ids() { # tag model prompt-name prompt-text spec arms...
  local tag=$1 model=$2 pn=$3 ptxt=$4 spec=$5; shift 5
  local SA=(); [ $spec = off ] && SA=(--spec off); [ $spec = mtp ] && SA=(--spec mtp)
  local ref="" ok=1 line=""
  for arm in "$@"; do
    run_arm $arm
    local o=$R/ids-$tag-$pn-$spec-$arm
    env $E $B generate --model $model --backend metal --max-seq 40000 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids $SA \
      --prompt "$ptxt" > $o.out 2> $o.err
    local got="$(grep -a '^ids' $o.err | md5 -q)" n=$(grep -a '^ids' $o.err | grep -oE '[0-9]+' | wc -l | tr -d ' ')
    local sp="$(grep -a 'spec:' $o.err | tail -1 | grep -oE 'passes=[0-9]+ proposed=[0-9]+ accepted=[0-9]+')"
    local sd="$(grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' $o.err | tail -1)"
    [ -z "$ref" ] && ref=$got; [ "$got" = "$ref" ] && [ $n -ge 200 ] || ok=0
    line="$line | $arm $got[1,8] n=$n $sp $sd"
  done
  echo "$( [ $ok = 1 ] && echo SAME || echo DIFFER ) $tag $pn spec=$spec$line"
}

case ${1:?step} in
  smoke)
    # the knob's two parts run at all: MTP default on both models, short + 4k, ids against the knob-off arm; writes the marker
    rm -f $R/SMOKE_OK
    out="$(ids 35b $M35 4k "$(cat $PR/prompt-4096.txt)" default off on; ids 27b $M27 short "$SHORT" default off on; ids 27b $M27 4k "$(cat $PR/prompt-4096.txt)" default off on)"
    echo "$out"
    [ "$(echo "$out" | grep -c '^SAME')" = 3 ] && touch $R/SMOKE_OK && echo "SMOKE OK"
    ;;
  confirm)
    # after merging main: main vs knob unset vs knob set, MTP default, both models
    ids 35b $M35 4k "$(cat $PR/prompt-4096.txt)" default base off on
    ids 35b $M35 32k "$(cat $PR/prompt-32768.txt)" default base off on
    ids 27b $M27 4k "$(cat $PR/prompt-4096.txt)" default base off on
    ids 27b $M27 short "$SHORT" off base off on
    ;;
  kernels)
    for arm in off on; do run_arm $arm
      env $E $B kernels test --backend metal > $R/kernels-$arm.out 2> $R/kernels-$arm.err; echo "kernels test $arm exit $? : $(tail -1 $R/kernels-$arm.out)"
      env $E $B kernels test --backend metal --model $M35 --op attn_decode > $R/kernels35-$arm.out 2>&1; echo "  35B attn_decode $arm exit $? : $(tail -1 $R/kernels35-$arm.out)"
      env $E $B kernels test --backend metal --model $M27 --op attn_decode > $R/kernels27-$arm.out 2>&1; echo "  27B attn_decode $arm exit $? : $(tail -1 $R/kernels27-$arm.out)"
      env $E $B kernels test --backend metal --model $M35 --op attn_prefill > $R/kernels35p-$arm.out 2>&1; echo "  35B attn_prefill $arm exit $? : $(tail -1 $R/kernels35p-$arm.out)"
    done
    grep -c FAIL $R/kernels-off.out $R/kernels-on.out
    diff <(grep -av 'us\b' $R/kernels-off.out | sed -E 's/[0-9.]+ ?(ms|us|s)\b//g') <(grep -av 'us\b' $R/kernels-on.out | sed -E 's/[0-9.]+ ?(ms|us|s)\b//g') > $R/kernels-diff.txt; echo "kernels test off vs on (times stripped): $(wc -l < $R/kernels-diff.txt) differing lines"
    ;;
  plan)
    for m in $M8 $M27 $M35; do for spec in default off mtp; do
      SA=(); [ $spec != default ] && SA=(--spec $spec)
      mn=${m:t}
      $BASE plan show --backend metal --model $m $SA --json > $R/plan-$mn-$spec-base.json 2> $R/plan-$mn-$spec-base.err; eb=$?
      $BIN  plan show --backend metal --model $m $SA --json > $R/plan-$mn-$spec-off.json 2> $R/plan-$mn-$spec-off.err; eo=$?
      INTARSIA_METAL_ATTN_KV_ONCE=1 $BIN plan show --backend metal --model $m $SA --json > $R/plan-$mn-$spec-on.json 2> $R/plan-$mn-$spec-on.err; en=$?
      echo "plan $mn spec=$spec: exits $eb/$eo/$en, base $(md5 -q $R/plan-$mn-$spec-base.json) off $(md5 -q $R/plan-$mn-$spec-off.json) on $(md5 -q $R/plan-$mn-$spec-on.json) $(cmp -s $R/plan-$mn-$spec-base.json $R/plan-$mn-$spec-off.json && echo 'base==off IDENTICAL' || echo 'base!=off DIFFER') (on vs off: $(diff $R/plan-$mn-$spec-off.json $R/plan-$mn-$spec-on.json | grep -c '^[<>]') lines)"
    done; done
    ;;
  ids35)
    [ -f $R/SMOKE_OK ] || { echo "no SMOKE_OK marker: skipped"; exit 3; }
    for spec in default off; do
      ids 35b $M35 short "$SHORT" $spec base off on
      ids 35b $M35 4k "$(cat $PR/prompt-4096.txt)" $spec base off on
      ids 35b $M35 4kb "$P4B" $spec base off on
      ids 35b $M35 16k "$(cat $PR/prompt-16384.txt)" $spec base off on
      ids 35b $M35 32k "$(cat $PR/prompt-32768.txt)" $spec base off on
    done ;;
  ids27)
    [ -f $R/SMOKE_OK ] || { echo "no SMOKE_OK marker: skipped"; exit 3; }
    for spec in default off; do
      ids 27b $M27 short "$SHORT" $spec base off on
      ids 27b $M27 4k "$(cat $PR/prompt-4096.txt)" $spec base off on
      ids 27b $M27 4kb "$P4B" $spec off on
      ids 27b $M27 16k "$(cat $PR/prompt-16384.txt)" $spec off on
      ids 27b $M27 32k "$(cat $PR/prompt-32768.txt)" $spec ${=ARMS32:-off on}
    done ;;
  ids8)
    for spec in default; do
      ids 8b $M8 short "$SHORT" $spec base off on
      ids 8b $M8 4k "$(cat $PR/prompt-4096.txt)" $spec base off on
    done ;;
esac
echo "## done $1"
