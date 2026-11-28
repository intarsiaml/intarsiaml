#!/bin/bash
# endor EXCLUSIVE fleetq slot A, lane ink-dflash-cut: the pre-registered A/B (docs/results/2026-11-27-inkling2-dflash-cut.md, "Plan").
# (1) ABBA 4 + 4 over speed.json (19 prompts, 256 tokens, EOS honoured), one process per arm@rep, 45 s GPU idle before each, macmon:
#     dfl5 = --spec dflash --spec-draft D --spec-k 5 (the Metal default since #551 / #553), cut7 = --spec-k 7 + INTARSIA_DFLASH_CUT=0.3.
# (2) greedy ids gate: off / dfl5 / cut7 / cut5 (k5 + cut 0.3) on g3.json (code, len4k, len16k), 256 tokens, --ignore-eos, md5 per prompt.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-dflash-cut; R=$W/docs/results/2026-11-27-inkling2-dflash-cut/raw; O=$R/A; mkdir -p $O
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
D=$HOME/.cache/huggingface/hub/models--incoai--Inkling-2-Mini-DFlash2/snapshots/4f9088ea66c21f2b99ed657d1941c306b03e84d2
B=$W/.scratch/bin/intarsia-ct
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
echo "slot A start $(date +%T) binary md5 $(md5 -q $B) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
T0=$(date +%s)
arm_flags() {
  case $1 in
    off) echo "--spec off" ;;
    dfl5) echo "--spec dflash --spec-draft $D --spec-k 5" ;;
    cut7) echo "--spec dflash --spec-draft $D --spec-k 7" ;;
    cut5) echo "--spec dflash --spec-draft $D --spec-k 5" ;;
  esac
}
arm_env() { case $1 in cut7|cut5) echo "INTARSIA_DFLASH_CUT=0.3" ;; *) : ;; esac; }
G="generate -m $M --backend metal --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256"
for tok in dfl5@r1 cut7@r1 cut7@r2 dfl5@r2 dfl5@r3 cut7@r3 cut7@r4 dfl5@r4; do
  arm=${tok%@*}; f=$O/$tok
  [ $(( $(date +%s) - T0 )) -ge 1500 ] && { echo "limit reached before $tok"; break; }
  sleep 45
  python3 -c 'import time; print(f"{time.time():.3f}")' > $f.t
  env $(arm_env $arm) $B $G --prompt-ids $R/speed.json $(arm_flags $arm) > $f.out 2> $f.err
  rc=$?
  python3 -c 'import time; print(f"{time.time():.3f}")' >> $f.t
  echo "$tok rc $rc $(date +%T) t+$(( $(date +%s) - T0 ))s | $(grep -aoE 'dflash: cut [0-9.]+' $f.err | head -1) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
done
echo "== ids gate (g3.json, --ignore-eos)"
for arm in off dfl5 cut7 cut5; do
  env $(arm_env $arm) $B $G --ignore-eos --prompt-ids $R/g3.json $(arm_flags $arm) > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? | $(grep -aE '^spec: verify' $O/g-$arm.err)"
done
python3 $R/md5.py $O/g-off.err $O/g-dfl5.err $O/g-cut7.err $O/g-cut5.err
echo "slot A end $(date +%T) t+$(( $(date +%s) - T0 ))s"
