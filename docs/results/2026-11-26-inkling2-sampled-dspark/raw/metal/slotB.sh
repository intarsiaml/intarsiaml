#!/bin/bash
# endor EXCLUSIVE fleetq slot, lane ink-sampled: Metal speed A/B, sampled decode with the recipe's DSpark default (no --spec), arm off
# (today: sampled requests take plain steps) vs arm on (INTARSIA_SPEC_SAMPLED_DFLASH=1), temperature 1.0 and 0.7 (top-k 0, top-p 1.0,
# seed 7: the same tokens in both arms, seed-exact), speed.json (warm-up, 3 short, 1k, 4k), 128 tokens each, --ignore-eos.
# Power protocol (docs/results/2026-11-25-inkling2-mode.md): 45 s GPU idle before every process, macmon pipe -i 250 for the slot.
# Reps alternate the arm order (odd: off1 on1 off07 on07, even: reversed). Time-boxed: no process starts after BUDGET_S.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-sampled; S=$W/.scratch; O=$S/metal/B-${1:-$(date +%m%d-%H%M)}; mkdir -p $O
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=$S/bin/intarsia-${BR:-br1}
BUDGET_S=${BUDGET_S:-1380}; IDLE=${IDLE:-45}; REPS=${REPS:-1 2 3 4}
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
echo "start $(date +%T) out $O binary md5 $(md5 -q $B)"
if [ -z "${SKIP_SEEDS:-}" ]; then
  # real-model seed equality on Metal (slot A's law run never reached its shared seeds) and proof that sampled verify rows take the
  # device sampler there: both arms on seeds 0..100 at T 1.0 / 0.6, the on arm with INTARSIA_SPEC_TRACE=1 ("keyed verify N rows (device
  # draws)"). Before macmon and the timed processes; the 45 s idle gap of the first timed process follows it.
  RT=$(ls -t $W/target/release/deps/inkling2_sampled_real-* | grep -v '\.d$' | head -1)
  for a in 0 1; do
    rm -f $O/seeds-$a.txt
    TR=INTARSIA_SPEC_TRACE=0; [ $a = 1 ] && TR=INTARSIA_SPEC_TRACE=1
    env INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=metal INTARSIA_SPEC_SAMPLED_DFLASH=$a \
      $TR INTARSIA_SS_SEEDS=0..${SEEDS_N:-100} INTARSIA_SS_TEMPS=1.0,0.6 INTARSIA_SS_TOKENS=6 \
      INTARSIA_SS_OUT=$O/seeds-$a.txt $RT --nocapture > $O/seeds-$a.log 2>&1
    echo "seeds arm $a rc $? $(date +%T) lines $(wc -l < $O/seeds-$a.txt) | $(grep -acE 'keyed verify .*device draws' $O/seeds-$a.log) device-draw passes, $(grep -acE 'keyed verify .*\(logits\)' $O/seeds-$a.log) logits passes"
  done
  cmp -s <(sort $O/seeds-0.txt) <(sort $O/seeds-1.txt) && echo "seeds: arms identical" || echo "seeds: ARMS DIFFER"
fi
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
T0=$(date +%s)
G="generate -m $M --backend metal --max-seq 8192 --profile-every 0 --show-ids --max-tokens ${TOKENS:-128} --ignore-eos --prompt-ids $S/prompts/speed.json --top-k 0 --top-p 1.0 --seed 7"
arm() { # name rep
  local f=$O/$1-r$2 on=0 t=1.0
  case $1 in on*) on=1 ;; esac
  case $1 in *07) t=0.7 ;; esac
  sleep $IDLE
  python3 -c 'import time; print(f"{time.time():.3f}")' > $f.t
  env ${TRACE:+INTARSIA_SPEC_TRACE=$TRACE} INTARSIA_SPEC_SAMPLED_DFLASH=$on $B $G --temperature $t > $f.out 2> $f.err
  local rc=$?
  python3 -c 'import time; print(f"{time.time():.3f}")' >> $f.t
  echo "$1 r$2 rc $rc $(date +%T) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
}
for rep in $REPS; do
  if [ $((rep % 2)) = 1 ]; then order="off1 on1 off07 on07"; else order="on07 off07 on1 off1"; fi
  for a in $order; do
    [ $(( $(date +%s) - T0 )) -ge $BUDGET_S ] && { echo "budget reached before $a r$rep"; break 2; }
    arm $a $rep
  done
done
echo "end $(date +%T)"
