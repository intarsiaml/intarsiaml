#!/usr/bin/env bash
# Window 2 (lane ink-hip-pf2-tiny; radiance down < 5 min): the fix's gates, two tracks at once.
#   card 0: `kernels test --backend hip --op ink_kv_prologue` (full sweep) on pf2b (case + fix); g1b 3 times on the branch's test
#           binary; the tiny `inkling2` suite (hip); greedy md5 + prefill, --spec off, short / 4k / 16k x 64 / 256, main3 (b7450223e)
#           and pf2b alternating (the 4k / 16k prefill pairs are same-card).
#   card 1: the same admission on pf2a (the new case on main's kernel: must FAIL there); greedy md5, spec default, the same matrix;
#           then pf2b with INTARSIA_HIP_POISON=ff (every new HIP allocation NaN-filled), short 64 spec off and 4k 256 spec default.
set -uo pipefail
L=$HOME/intarsiaml-ink-hip-pf2-tiny
R=$L/docs/results/2026-11-26-inkling2-hip-pf2-tiny/raw
OUT=$L/logs/w2
mkdir -p "$OUT"
cp -p "$HOME/intarsiaml-ink-hip-p12/.scratch/bin/intarsia-main3" "$L/.scratch/bin/intarsia-main3"
M=/models/Inkling-2-Mini
PS="$(cat "$L/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$L/scripts/cmx/ids-prompts/4k-0.txt")"
P16="$P4
$P4
$P4
$P4"
now() { date '+%T'; }
pr() { case $1 in short) echo "$PS" ;; 4k) echo "$P4" ;; 16k) echo "$P16" ;; esac; }
gen() {  # card arm binary env spec prompt-name tokens
  local f=$OUT/c$1-$2-$5-$6-$7 sp="" ms=8192
  [ "$5" = off ] && sp="--spec off"
  [ "$6" = 16k ] && ms=20480
  CARD=$1 bash "$R/run.sh" env $4 /work/.scratch/bin/intarsia-$3 generate -m $M --backend hip --max-seq $ms $sp --ignore-eos --profile-every 0 --max-tokens "$7" -p "$(pr $6)" > "$f.out" 2> "$f.err"
  echo "MD5 c$1 $2 $5-$6 $7 $(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) [$(now)]"
}
adm() {  # card binary
  CARD=$1 bash "$R/run.sh" /work/.scratch/bin/intarsia-$2 kernels test --backend hip --op ink_kv_prologue --verbose > "$OUT/adm-$2.log" 2>&1
  echo "ADM $2: $(grep -acE 'FAIL' "$OUT/adm-$2.log") FAIL lines | dr=16 E=12: $(grep -ac 'dr=16 E=12' "$OUT/adm-$2.log") lines, $(grep -a 'dr=16 E=12' "$OUT/adm-$2.log" | grep -aE 'FAIL' | grep -ac 'T=40') FAIL at T=40, $(grep -a 'FAIL' "$OUT/adm-$2.log" | grep -avc 'dr=16 E=12') FAIL elsewhere | $(grep -aE 'ADMITTED|REJECT' "$OUT/adm-$2.log" | tail -2 | tr '\n' ' ' | cut -c1-240) [$(now)]"
  grep -aE 'FAIL' "$OUT/adm-$2.log" | head -3 | cut -c1-220
}
card1() {
  adm 1 pf2a
  for p in short 4k 16k; do
    gen 1 main main3 X=1 default $p 64; gen 1 branch pf2b X=1 default $p 64
    gen 1 branch pf2b X=1 default $p 256; gen 1 main main3 X=1 default $p 256
  done
  gen 1 poison pf2b INTARSIA_HIP_POISON=ff off short 64
  gen 1 poison pf2b INTARSIA_HIP_POISON=ff default 4k 256
  echo "[card1] done $(now)"
}
card1 > "$OUT/card1.log" 2>&1 &
C1=$!
echo "[stage] card 0 start $(now)"
adm 0 pf2b
for i in 1 2 3; do
  echo "G1B r$i $(CARD=0 bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test inkling2 inkling2_tiny_g1b_teacher_forced_bf16 -- --exact --test-threads 1 2>&1 | grep -aoE 'test inkling2_tiny_g1b_teacher_forced_bf16 \.\.\. [A-Za-z]+|rel err [^ ]+ > budget [^ ]+' | head -2 | tr '\n' ' ') [$(now)]"
done
echo "[stage] tiny inkling2 suite on hip, pf2b $(now)"
CARD=0 bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test inkling2 -- --test-threads 1 2>&1 | grep -aE "FAILED|failed|test result" | head -8
for p in short 4k 16k; do
  gen 0 main main3 X=1 off $p 64; gen 0 branch pf2b X=1 off $p 64
  gen 0 branch pf2b X=1 off $p 256; gen 0 main main3 X=1 off $p 256
done
echo "[stage] card 0 done $(now); waiting for card 1"
wait $C1
cat "$OUT/card1.log"
echo "[stage] done $(now) (${SECONDS}s)"
