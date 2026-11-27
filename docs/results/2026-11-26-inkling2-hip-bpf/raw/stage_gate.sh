#!/usr/bin/env bash
# Window stage (lane ink-hip-bpf), inside window.sh: gate 3, the solo path unchanged. Default env (no INTARSIA_* knob), graphs.
# Greedy md5 of `generate` at 64 and 256 tokens, short and 4k prompts, main (80719483a) then the branch per case: card 0 with
# --spec off, card 1 at the same time with the DSpark default (each arm pair on one card). G2 and admission: stage_g2.sh.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
OUT=$REPO/logs/gate
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
MA=${MA:-main}
BR=${BR:-b3}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
now() { date '+%F %T'; }
gen() {  # card arm binary prompt-name prompt tokens spec-args
  local card=$1; shift
  local tag=$1-$3-$5-${6:-dspark}; tag=${tag// /}
  local f=$OUT/$tag
  local cn=ink-hip-bpf-run; [ "$card" = 1 ] && cn=ink-hip-bpf-run2
  HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$cn bash "$S/run.sh" /work/.scratch/bin/intarsia-$2 generate -m $M --backend hip --max-seq 8192 $6 --ignore-eos \
    --profile-every 0 --max-tokens "$5" -p "$4" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 card$card $1 $3 $5 ${6:-dspark} $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) [$(now)]"
}
md5s() {  # card spec-args
  for p in short 4k; do
    [ $p = short ] && PR="$PS" || PR="$P4"
    for n in 64 256; do
      gen $1 main $MA $p "$PR" $n "$2"
      gen $1 branch $BR $p "$PR" $n "$2"
    done
  done
  echo "[stage] card$1 done $(now)"
}
md5s 0 "--spec off" &
md5s 1 "" &
wait
echo "[stage] done $(now)"
