#!/usr/bin/env bash
# Lane ink-hip-specflip, window D1 (the sampled DFlash2 test, step 2: calibration), ON tatooine inside window.sh. Binary $BR (default
# specA1: sampled DFlash-family drafting with INTARSIA_SPEC_SAMPLED_DFLASH=1 and any --spec-draft), both cards at once, one process at
# a time per card, #543's W1 calibration protocol: speed-adapt.json (#543's speed.json, md5 0c1a62af...), 128 tokens, --ignore-eos,
# --top-k 0 --top-p 1.0 --seed 11, INTARSIA_SPEC_TRACE=1, the DFlash2 drafter at k4: mode 0 (fixed k4, INTARSIA_SPEC_SAMPLED_ADAPT=0)
# then `drafts`. Card 0 T 1.0, card 1 T 0.7. Then card 0 (card 1 still timing or done; no timed run shares card 0): 100 seeds at T 1.0,
# 24 tokens, k4, DFlash2: drafting off (=0) vs on (=1, the recipe's sampled_adapt string); lines must be identical.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
BR=${BR:-specA1}
OUT=$REPO/logs/d1; mkdir -p "$OUT/cal" "$REPO/.scratch/prompts" "$REPO/.scratch/home"
cp "$RAW/speed-adapt.json" "$REPO/.scratch/prompts/"
M=/models/Inkling-2-Mini; D=/models/Inkling-2-Mini-DFlash2
B=/work/.scratch/bin/intarsia-$BR
now() { date '+%T'; }
echo "[d1] start $(now) $BR $(md5sum "$REPO/.scratch/bin/intarsia-$BR" | cut -c1-32) prompts $(md5sum "$RAW/speed-adapt.json" | cut -c1-32)"
rc() { local c=$1; shift; env CARD=$c CNAME=ink-hip-specflip-run$c bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
G="generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --spec dflash --spec-draft $D --spec-k 4"
SP="--max-tokens 128 --ignore-eos --prompt-ids /work/.scratch/prompts/speed-adapt.json --top-k 0 --top-p 1.0 --seed 11"
cal() { # card temp mode
  local f=$OUT/cal/$3-t$2
  rc $1 env INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=$3 INTARSIA_SPEC_TRACE=1 $B $G $SP --temperature $2 > $f.out 2> $f.err
  echo "[d1 c$1] cal $3 T $2 rc $? $(now) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err) | $(grep -aE 'DFlash-family drafting|draft length per request' $f.err | tr '\n' ' ' | cut -c1-260)"
}
seeds() { # card arm(0|1)
  local f=$OUT/seeds-$2
  rm -f $f.txt
  rc $1 env INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_SS_DRAFT=$D INTARSIA_INKLING2_BACKEND=hip INTARSIA_SPEC_SAMPLED_DFLASH=$2 \
    INTARSIA_SS_SEEDS=0..100 INTARSIA_SS_TEMPS=1.0 INTARSIA_SS_TOKENS=24 INTARSIA_SS_K=4 INTARSIA_SS_BUDGET_S=${SEEDS_BUDGET:-75} \
    INTARSIA_SS_OUT=/work/logs/d1/seeds-$2.txt /work/.scratch/bin/t-$BR-inkling2_sampled_real --nocapture > $f.log 2>&1
  echo "[d1 c$1] seeds arm $2 rc $? $(now) lines $(wc -l < $f.txt 2>/dev/null) | $(grep -aE 'requests in|draft of|DFlash-family drafting|draft length per request|draft Some' $f.log | tr '\n' ' ' | cut -c1-500)"
}
( cal 0 1.0 0; cal 0 1.0 drafts; seeds 0 0; seeds 0 1 ) > "$OUT/c0.log" 2>&1 &
( cal 1 0.7 0; cal 1 0.7 drafts ) > "$OUT/c1.log" 2>&1 &
wait
cat "$OUT/c0.log" "$OUT/c1.log"
if [ -s "$OUT/seeds-0.txt" ] && [ -s "$OUT/seeds-1.txt" ]; then
  a=$(sort "$OUT/seeds-0.txt"); b=$(sort "$OUT/seeds-1.txt")
  echo "[d1] seeds T 1.0 (DFlash2 k4): off $(echo "$a" | grep -c .) on $(echo "$b" | grep -c .) shared $(comm -12 <(echo "$a" | cut -d' ' -f1,2) <(echo "$b" | cut -d' ' -f1,2) | grep -c .) identical $(comm -12 <(echo "$a") <(echo "$b") | grep -c .)"
fi
echo "[d1] done $(now)"
