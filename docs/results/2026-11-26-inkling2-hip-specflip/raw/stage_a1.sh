#!/usr/bin/env bash
# Lane ink-hip-specflip, option (a) window WA1 (pre-registered gate in the results doc, 00:25 EST 11-27), ON tatooine inside window.sh.
#   BR  = the (a) binary (both drafts by default), CTL = PR A's binary (specA2, one drafter).
# 1. KV first, card 1: solo `generate --max-seq 1000000` (the context printed = the memory-limited KV capacity): CTL default (DSpark
#    alone) and BR default (DSpark + DFlash2); then BR at the defaults (`--max-seq` 4096). Gate 3: BR capacity >= 0.90 x CTL capacity
#    (stop line -10 %), else the speed part is skipped (file $OUT/KV_FAIL).
# 2. Card 0 (after the KV verdict): greedy ABBA half (stage_ab_k.sh's prompts and protocol: speed.json, 256 tokens, greedy, EOS honoured):
#    arms dspark = CTL default (DSpark k5, main's and PR A's greedy default), dual = BR default (coordinator 00:43 change 1: the GO rule vs
#    DSpark k5); ORDER a = dspark dual dual dspark (reps R1 R2), b = dual dspark dspark dual. Out: logs/a-greedy/<arm>@<rep>.err
#    (speed.py <dir> dspark <classes>).
# 3. Card 1 (after the KV), the tasks in $C1 (default "md5 seeds1.0"; WA2: KV=0 C1="seeds0.6 md564"): md5 = greedy md5 (md5.json, 256
#    tokens, --ignore-eos) BR --spec off / BR default / CTL --spec dflash <DFlash2> k4 (change 6: acceptance counts); md564 = BR default at 64 tokens (vs the first 64 ids of md5's off run, same
#    folder); seeds<T> = 100 seeds (24 tokens) at T: BR with INTARSIA_SPEC_SAMPLED_DFLASH=0 vs BR default (INTARSIA_SS_DEFAULT=1: no
#    --spec, the recipe default with both drafts); rev<T> = the same seeds with INTARSIA_SPEC_DFLASH2_DEFAULT=0 (PR A's load, DSpark alone:
#    change 6's sampled acceptance comparison, the same harness path and binary); mixed = change 7 (8 requests alternating greedy / seeded sampled in one process,
#    prompt x1 and x12; BR default vs INTARSIA_SPEC_DEFAULT=0).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
S=$REPO/.scratch
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
M=/models/Inkling-2-Mini
D=/models/Inkling-2-Mini-DFlash2
BR=${BR:?set BR}; CTL=${CTL:-specA2}; ORDER=${ORDER:-a}; R1=${R1:-r1}; R2=${R2:-r2}; KV=${KV:-1}; C1=${C1:-md5 seeds1.0}
OUT=$REPO/logs/a1-$BR; G=$REPO/logs/a-greedy; mkdir -p "$OUT" "$G" "$S/prompts" "$S/home"
cp "$RAW/speed.json" "$RAW/md5.json" "$S/prompts/"
B=/work/.scratch/bin/intarsia-$BR; C=/work/.scratch/bin/intarsia-$CTL
for x in $BR $CTL; do [ -x "$S/bin/intarsia-$x" ] || { echo "[a1] guard: $x missing"; exit 3; }; done
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
now() { date '+%T'; }
rc() { local c=$1; shift; env CARD=$c CNAME=ink-hip-specflip-run$c bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
echo "[a1] start $(now) BR $BR $(md5sum "$S/bin/intarsia-$BR" | cut -c1-32) $(cat "$S/bin/intarsia-$BR.commit" 2>/dev/null) CTL $CTL $(md5sum "$S/bin/intarsia-$CTL" | cut -c1-32) ORDER $ORDER $R1 $R2"
kvl() { grep -aE '^loaded|^dflash: |DSpark default|DFlash2 drafter|^Error' "$1" | tr '\n' ' ' | cut -c1-700; }
cap() { grep -aoE 'context [0-9]+ tokens' "$1" | head -1 | grep -oE '[0-9]+'; }
if [ "$KV" = 1 ]; then
  kvgen() {  # name binary extra
    rc 1 $2 generate -m $M --backend hip --max-tokens 1 --prompt Hello $3 > "$OUT/kv-$1.out" 2> "$OUT/kv-$1.err"
    echo "[a1 kv] $1 rc $? t+$(el)s | $(kvl "$OUT/kv-$1.err")"
  }
  kvgen ctl-big $C "--max-seq 1000000"; kvgen dual-big $B "--max-seq 1000000"; kvgen dual-default $B ""
  c0=$(cap "$OUT/kv-ctl-big.err"); c1=$(cap "$OUT/kv-dual-big.err"); cd=$(cap "$OUT/kv-dual-default.err")
  if [ -n "$c0" ] && [ -n "$c1" ]; then
    echo "[a1 kv] capacity DSpark alone $c0, both drafts $c1: $(python3 -c "print(f'{($c1 - $c0) / $c0 * 100:+.2f} %')"); default context $cd"
    python3 -c "import sys; sys.exit(0 if $c1 >= 0.90 * $c0 else 1)" || { echo "[a1 kv] GATE 3 FAILS (worse than -10 %): speed part skipped"; touch "$OUT/KV_FAIL"; }
    [ "$cd" = 4096 ] || { echo "[a1 kv] GATE 3 FAILS: the default solo context is $cd, not 4096"; touch "$OUT/KV_FAIL"; }
  else
    echo "[a1 kv] no capacity line: speed part skipped"; touch "$OUT/KV_FAIL"
  fi
fi
GEN="generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256 --prompt-ids /work/.scratch/prompts/speed.json"
arm() {  # arm rep
  local f=$G/$1@$2
  if [ "$1" = dspark ]; then rc 0 $C $GEN > "$f.out" 2> "$f.err"
  else rc 0 $B $GEN > "$f.out" 2> "$f.err"; fi
  echo "[a1 c0] $1@$2 rc $? t+$(el)s | $(grep -aoE 'up to [0-9]+ drafts' "$f.err" | head -1) | $(grep -aE '^decode overall' "$f.err" | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100) | $(grep -aE 'DFlash2 drafter|second draft' "$f.err" | tr '\n' ' ' | cut -c1-200)"
}
speed() {
  [ -e "$OUT/KV_FAIL" ] && { echo "[a1 c0] skipped (KV)"; return; }
  if [ "$ORDER" = a ]; then arm dspark $R1; arm dual $R1; arm dual $R2; arm dspark $R2
  else arm dual $R1; arm dspark $R1; arm dspark $R2; arm dual $R2; fi
}
md5() {  # name tokens extra [binary]
  local f=$OUT/md5-$1
  rc 1 ${4:-$B} generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --ignore-eos --max-tokens $2 --prompt-ids /work/.scratch/prompts/md5.json $3 > "$f.out" 2> "$f.err"
  echo "[a1 c1] md5 $1 rc $? t+$(el)s $(cat "$f.out" "$f.err" | grep -a '^ids\[' | md5sum | cut -c1-32) | $(grep -aE '^decode overall|^spec: verify|DFlash2 drafter|second draft|DSpark default' "$f.err" | tr '\n' ' ' | cut -c1-420)"
}
seeds() {  # arm (off | def) temp
  local f=$OUT/seeds-$1-$2 e="INTARSIA_SS_DEFAULT=1"
  rm -f "$f.txt"
  [ "$1" = off ] && e="$e INTARSIA_SPEC_SAMPLED_DFLASH=0"
  [ "$1" = rev ] && e="$e INTARSIA_SPEC_DFLASH2_DEFAULT=0"
  rc 1 env $e INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=hip INTARSIA_SS_SEEDS=0..100 INTARSIA_SS_TEMPS=$2 \
    INTARSIA_SS_TOKENS=24 INTARSIA_SS_BUDGET_S=75 INTARSIA_SS_OUT=/work/logs/a1-$BR/seeds-$1-$2.txt \
    /work/.scratch/bin/t-$BR-inkling2_sampled_real --nocapture > "$f.log" 2>&1
  echo "[a1 c1] seeds $1 T $2 rc $? t+$(el)s lines $(wc -l < "$f.txt" 2>/dev/null) | $(grep -aE 'requests in|DFlash2 drafter|DFlash-family drafting|second draft' "$f.log" | tr '\n' ' ' | cut -c1-400)"
  [ "$1" = def ] && ! grep -aq "DFlash2 drafter .*loaded beside" "$f.log" && echo "[a1 c1] seeds def T $2: VOID (no DFlash2 drafter line: the recipe default did not apply)"
}
mixed() {  # arm (off = no speculation at all | def) prompt-repeat: greedy (T 0) and seeded sampled (T 1.0) alternating, seeds 0..4
  local f=$OUT/mixed-$1-x$2 e="INTARSIA_SS_DEFAULT=1"
  rm -f "$f.txt"
  [ "$1" = off ] && e="$e INTARSIA_SPEC_DEFAULT=0"
  rc 1 env $e INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=hip INTARSIA_SS_SEEDS=0..4 INTARSIA_SS_TEMPS=0.0,1.0 \
    INTARSIA_SS_TOKENS=24 INTARSIA_SS_PROMPT_X=$2 INTARSIA_SS_OUT=/work/logs/a1-$BR/mixed-$1-x$2.txt \
    /work/.scratch/bin/t-$BR-inkling2_sampled_real --nocapture > "$f.log" 2>&1
  echo "[a1 c1] mixed $1 x$2 rc $? t+$(el)s lines $(wc -l < "$f.txt" 2>/dev/null) | $(grep -aE 'requests in|DFlash2 drafter|panicked' "$f.log" | tr '\n' ' ' | cut -c1-400)"
}
cmp_mixed() {  # prompt-repeat
  [ -s "$OUT/mixed-off-x$1.txt" ] && [ -s "$OUT/mixed-def-x$1.txt" ] || { echo "[a1] mixed x$1: missing"; return; }
  echo "[a1] mixed x$1: off $(grep -c . "$OUT/mixed-off-x$1.txt") default $(grep -c . "$OUT/mixed-def-x$1.txt") $(cmp -s "$OUT/mixed-off-x$1.txt" "$OUT/mixed-def-x$1.txt" && echo IDENTICAL || echo DIFFER)"
}
cmp_seeds() {  # temp
  local a b
  [ -s "$OUT/seeds-off-$1.txt" ] && [ -s "$OUT/seeds-def-$1.txt" ] || { echo "[a1] seeds T $1: missing"; return; }
  a=$(sort "$OUT/seeds-off-$1.txt"); b=$(sort "$OUT/seeds-def-$1.txt")
  echo "[a1] seeds T $1: off $(echo "$a" | grep -c .) default $(echo "$b" | grep -c .) identical $(comm -12 <(echo "$a") <(echo "$b") | grep -c .)"
}
speed > "$OUT/c0.log" 2>&1 &
(
  for t in $C1; do
    case $t in
      md5) md5 off 256 "--spec off"; md5 def 256 ""; md5 dfl4 256 "--spec dflash --spec-draft $D --spec-k 4" $C ;;
      md564) md5 def64 64 "" ;;
      seeds*) seeds off ${t#seeds}; seeds def ${t#seeds} ;;
      rev*) seeds rev ${t#rev} ;;
      mixed) mixed off 1; mixed def 1; mixed off 12; mixed def 12 ;;
    esac
  done
) > "$OUT/c1.log" 2>&1 &
wait
cat "$OUT/c0.log" "$OUT/c1.log"
if [ -s "$OUT/md5-def.err" ] && [ -s "$OUT/md5-off.err" ]; then
  o=$(cat "$OUT/md5-off.out" "$OUT/md5-off.err" | grep -a '^ids\[' | md5sum | cut -c1-32); d=$(cat "$OUT/md5-def.out" "$OUT/md5-def.err" | grep -a '^ids\[' | md5sum | cut -c1-32)
  echo "[a1] greedy md5 256: off $o default $d $([ "$o" = "$d" ] && echo EQUAL || echo DIFFER)"
fi
[ -s "$OUT/md5-def64.err" ] && [ -s "$OUT/md5-off.err" ] && python3 - "$OUT" <<'EOF'
import sys, re
from pathlib import Path
d = Path(sys.argv[1])
def ids(p):
    out = {}
    for l in (p.with_suffix(".out").read_text(errors="replace") + p.with_suffix(".err").read_text(errors="replace")).splitlines():
        m = re.match(r"^ids\[([^\]]*)\]: (.*)$", l)
        if m:
            out[m.group(1)] = m.group(2).split()
    return out
off, d64 = ids(d / "md5-off.err"), ids(d / "md5-def64.err")
eq = all(off.get(k, [])[:len(v)] == v and len(v) == 64 for k, v in d64.items()) and len(d64) == len(off) > 0
print(f"[a1] greedy 64 tokens: default's ids = the first 64 of --spec off's ({len(d64)} prompts): {'EQUAL' if eq else 'DIFFER'}")
EOF
for t in $C1; do case $t in seeds*) cmp_seeds ${t#seeds} ;; mixed) cmp_mixed 1; cmp_mixed 12 ;; esac; done
# change 6: acceptance counts, dual default vs DFlash2 k4 alone on md5.json
if [ -s "$OUT/md5-def.err" ] && [ -s "$OUT/md5-dfl4.err" ]; then
  a=$(grep -aE '^spec(\[[^]]*\])?: verify' "$OUT/md5-def.err" | md5sum | cut -c1-12); b=$(grep -aE '^spec(\[[^]]*\])?: verify' "$OUT/md5-dfl4.err" | md5sum | cut -c1-12)
  echo "[a1] acceptance md5.json: dual $(grep -aE '^spec: verify' "$OUT/md5-def.err" | cut -c1-80) | DFlash2 k4 alone $(grep -aE '^spec: verify' "$OUT/md5-dfl4.err" | cut -c1-80) | spec lines $a / $b $([ "$a" = "$b" ] && echo EQUAL || echo DIFFER)"
fi
for t in $C1; do case $t in rev*) T=${t#rev}
  a=$(grep -aE 'requests in' "$OUT/seeds-def-$T.log" | grep -oE 'spec verify.*'); b=$(grep -aE 'requests in' "$OUT/seeds-rev-$T.log" | grep -oE 'spec verify.*')
  echo "[a1] acceptance seeds T $T: dual [$a] vs DSpark alone (INTARSIA_SPEC_DFLASH2_DEFAULT=0, PR A's load) [$b] $([ -n "$a" ] && [ "$a" = "$b" ] && echo EQUAL || echo DIFFER); ids def vs rev: $(cmp -s <(sort "$OUT/seeds-def-$T.txt") <(sort "$OUT/seeds-rev-$T.txt") && echo IDENTICAL || echo DIFFER)" ;; esac; done
echo "[a1] done $(now) t+$(el)s"
