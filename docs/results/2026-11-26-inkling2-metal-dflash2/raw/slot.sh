#!/bin/bash
# endor EXCLUSIVE fleetq slot, lane ink-metal-dflash2: Metal runs of `generate` on Inkling-2-Mini, one process per "arm@rep" token, in the
# order given (ARMS="dspark@r1 dfl4@r1 dfl4@r2 dspark@r2"). Arms: dspark = no --spec flags (main's Metal default, DSpark k3), off = --spec off,
# dflK = --spec dflash --spec-draft $D --spec-k K (the DFlash2 drafter). PROMPTS = speed (speed.json, 256 tokens, EOS honoured, as #541)
# or md5 (md5.json, --ignore-eos, TOK 64 / 256). Power protocol (docs/results/2026-11-25-inkling2-mode.md, as #535's slotB.sh): IDLE s of GPU
# idle before every process, macmon pipe -i 250 for the slot. No process starts after LIMIT s. Out: $O/<arm>@<rep>.{err,out,t}.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-metal-dflash2; S=$W/.scratch
O=$S/metal/${NAME:?NAME}; mkdir -p $O
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
D=$HOME/.cache/huggingface/hub/models--incoai--Inkling-2-Mini-DFlash2/snapshots/4f9088ea66c21f2b99ed657d1941c306b03e84d2
B=$S/bin/intarsia-${BR:-br1}
PROMPTS=${PROMPTS:-speed}; TOK=${TOK:-256}; LIMIT=${LIMIT:-1080}; IDLE=${IDLE:-45}
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
echo "start $(date +%T) out $O binary md5 $(md5 -q $B) git $(git -C $W rev-parse --short HEAD) prompts $PROMPTS tok $TOK idle $IDLE limit $LIMIT"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
T0=$(date +%s)
X=""; [ "$PROMPTS" = md5 ] && X="--ignore-eos"
G="generate -m $M --backend metal --max-seq 20480 --profile-every 0 --show-ids --max-tokens $TOK $X --prompt-ids $S/prompts/$PROMPTS.json"
for tok in $ARMS; do
  arm=${tok%@*}; f=$O/$tok
  [ $(( $(date +%s) - T0 )) -ge $LIMIT ] && { echo "limit reached before $tok"; break; }
  case $arm in
    dspark) F="" ;;
    off) F="--spec off" ;;
    dfl*) F="--spec dflash --spec-draft $D --spec-k ${arm#dfl}" ;;
    *) echo "bad arm $arm"; continue ;;
  esac
  sleep $IDLE
  python3 -c 'import time; print(f"{time.time():.3f}")' > $f.t
  $B $G $F > $f.out 2> $f.err
  rc=$?
  python3 -c 'import time; print(f"{time.time():.3f}")' >> $f.t
  echo "$tok rc $rc $(date +%T) t+$(( $(date +%s) - T0 ))s | $(grep -aoE 'up to [0-9]+ drafts' $f.err | head -1) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
done
echo "end $(date +%T)"
