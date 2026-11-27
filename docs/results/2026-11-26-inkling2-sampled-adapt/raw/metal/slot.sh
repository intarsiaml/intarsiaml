#!/bin/bash
# endor EXCLUSIVE fleetq slot (<= 30 min), lane ink-sampled-adapt: Metal (recipe default DSpark k3, cut 0.5, device sampler for
# plain-temperature requests). Env: SIMSET (non-table parameters chosen on HIP, sim.py --set=), ADAPT_FORCE (the policy arm's string).
# 1. seed equality (real model, ad4, drafting off vs the policy, seeds 0..100 x T 1.0 / 0.6, 24 tokens, k3);
# 2. the tiny adapt suite on Metal (SKIP_TINY=1 skips it); 3. macmon; traced fixed-k3 and `drafts` calibration processes (seed 11,
# T 1.0), the 15 % cost-table rule and the replay (sim.py --metal: the grid only if the defaults miss the 0.99 bars); 4. speed: arms off (ad4, sampled drafting off), fix (main6 = main aee06201c, INTARSIA_SPEC_SAMPLED_DFLASH=1), pol
# (ad4, =1, + the measured table if the rule says so), T 1.0 and 0.7, speed.json, 128 tokens, --ignore-eos --top-k 0 --top-p 1.0
# --seed 7; 3 reps, Latin square per rep (off fix pol / fix pol off / pol off fix), both temperatures per rep; 45 s GPU idle before every
# timed process. Time-boxed: no process starts after BUDGET_S.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-sampled-adapt; S=$W/.scratch; O=$S/metal/run-${1:-$(date +%m%d-%H%M)}; mkdir -p $O/speed $O/cal
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
BUDGET_S=${BUDGET_S:-1650}; IDLE=${IDLE:-45}
T0=$(date +%s)
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
echo "start $(date +%T) out $O | $(md5 -q $S/bin/intarsia-ad4) ad4 | $(md5 -q $S/bin/intarsia-main6) main6"
RT=$(ls -t $W/target/release/deps/inkling2_sampled_real-* | grep -v '\.d$' | head -1)
TT=$(ls -t $W/target/release/deps/inkling2_dspark_sampled_adapt-* | grep -v '\.d$' | head -1)
if [ -z "${SKIP_SEEDS:-}" ]; then
  for a in 0 1; do
    rm -f $O/seeds-$a.txt
    TR=INTARSIA_SPEC_TRACE=0; [ $a = 1 ] && TR=INTARSIA_SPEC_TRACE=1
    env INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=metal INTARSIA_SPEC_SAMPLED_DFLASH=$a \
      $TR INTARSIA_SS_SEEDS=0..100 INTARSIA_SS_TEMPS=1.0,0.6 INTARSIA_SS_TOKENS=24 INTARSIA_SS_K=3 INTARSIA_SS_BUDGET_S=110 \
      INTARSIA_SS_OUT=$O/seeds-$a.txt $RT --nocapture > $O/seeds-$a.log 2>&1
    echo "seeds arm $a rc $? $(date +%T) lines $(wc -l < $O/seeds-$a.txt) | $(grep -acE 'keyed verify .*device draws' $O/seeds-$a.log) device-draw passes, $(grep -acE 'keyed verify .*\(logits\)' $O/seeds-$a.log) logits passes | $(grep -ac 'Stop {' $O/seeds-$a.log) stops, $(grep -ac 'Resume {' $O/seeds-$a.log) resumes | $(grep -aE 'requests in' $O/seeds-$a.log)"
  done
  for t in 1 0.6; do
    x=$(awk -v t=$t '$1==t' $O/seeds-0.txt | sort); y=$(awk -v t=$t '$1==t' $O/seeds-1.txt | sort)
    echo "seeds T $t: off $(echo "$x" | grep -c .) on $(echo "$y" | grep -c .) shared $(comm -12 <(echo "$x" | cut -d' ' -f1,2) <(echo "$y" | cut -d' ' -f1,2) | grep -c .) identical $(comm -12 <(echo "$x") <(echo "$y") | grep -c .)"
  done
fi
if [ -z "${SKIP_TINY:-}" ]; then
  env INTARSIA_INKLING2_BACKEND=metal $TT --test-threads=1 --nocapture > $O/tiny-adapt-metal.txt 2>&1
  echo "tiny inkling2_dspark_sampled_adapt (metal) rc $? $(date +%T) | $(grep -aE '^test result' $O/tiny-adapt-metal.txt)"
fi
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
G="generate -m $M --backend metal --max-seq 8192 --profile-every 0 --show-ids --max-tokens 128 --ignore-eos --prompt-ids $S/prompts/speed.json --top-k 0 --top-p 1.0"
stamp() { python3 -c 'import time; print(f"{time.time():.3f}")'; }
sleep $IDLE
env INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=0 INTARSIA_SPEC_TRACE=1 $S/bin/intarsia-ad4 $G --seed 11 --temperature 1.0 > $O/cal/0-t1.0.out 2> $O/cal/0-t1.0.err
echo "cal fixed k3 T 1.0 rc $? $(date +%T) | $(grep -aE '^decode overall' $O/cal/0-t1.0.err) | $(grep -aE '^spec: verify' $O/cal/0-t1.0.err)"
sleep $IDLE
env INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=drafts INTARSIA_SPEC_TRACE=1 $S/bin/intarsia-ad4 $G --seed 11 --temperature 1.0 > $O/cal/drafts-t1.0.out 2> $O/cal/drafts-t1.0.err
echo "cal drafts T 1.0 rc $? $(date +%T) | $(grep -aE '^decode overall' $O/cal/drafts-t1.0.err)"
python3 $S/sim.py $O/cal --metal ${SIMSET:+--set=$SIMSET} > $O/cal/sim.txt 2>&1
grep -vE '^  g [0-9]' $O/cal/sim.txt | head -60
ADAPT=${ADAPT_FORCE:-$(grep -oE '^adapt string: .*' $O/cal/sim.txt | sed 's/^adapt string: //')}
echo "policy arm: ${ADAPT:+INTARSIA_SPEC_SAMPLED_ADAPT=$ADAPT}${ADAPT:-the built-in table}"
arm() { # name rep
  local f=$O/speed/$1-r$2 t=1.0 bin=ad4 pre="-u INTARSIA_SPEC_SAMPLED_DFLASH"
  case $1 in fix*) bin=main6; pre="INTARSIA_SPEC_SAMPLED_DFLASH=1" ;; pol*) pre="INTARSIA_SPEC_SAMPLED_DFLASH=1 ${ADAPT:+INTARSIA_SPEC_SAMPLED_ADAPT=$ADAPT}" ;; esac
  case $1 in *07) t=0.7 ;; esac
  sleep $IDLE
  stamp > $f.t
  env $pre $S/bin/intarsia-$bin $G --seed 7 --temperature $t > $f.out 2> $f.err
  local rc=$?
  stamp >> $f.t
  echo "$1 r$2 rc $rc $(date +%T) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
}
for rep in 1 2 3; do
  case $rep in 1) order="off fix pol" ;; 2) order="fix pol off" ;; 3) order="pol off fix" ;; esac
  for tt in 1 07; do
    for a in $order; do
      [ $(( $(date +%s) - T0 )) -ge $BUDGET_S ] && { echo "budget reached before $a$tt r$rep"; break 3; }
      arm $a$tt $rep
    done
  done
done
echo "end $(date +%T) ($(( $(date +%s) - T0 )) s)"
python3 $S/speed3.py $O/speed --macmon > $O/speed/speed.txt 2>&1
cat $O/speed/speed.txt
