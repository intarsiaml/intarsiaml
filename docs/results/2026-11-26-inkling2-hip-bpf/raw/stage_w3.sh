#!/usr/bin/env bash
# Window stage W3 (lane ink-hip-bpf), inside window.sh: gate 3 (the solo path unchanged) and the admission prime of the next windows'
# servers. Default env (no INTARSIA_* knob unless named), graphs.
#   card 0: prime the branch's packed batching server (admission cache, HOME=.scratch/home, one file per binary), then greedy md5 of
#           `generate` at 64 / 256 tokens, short-0 / 4k-0, --spec off: main ($MA) then the branch ($BR) per case.
#   card 1: prime main's batching server, then the same md5 cases with the DSpark default (no --spec).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
OUT=$REPO/logs/w3
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
MA=${MA:-main2}
BR=${BR:-b8}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
now() { date '+%F %T'; }
cname() { [ "$1" = 1 ] && echo ink-hip-bpf-run2 || echo ink-hip-bpf-run; }
gen() {  # card binary prompt-name tokens spec-args
  local card=$1 b=$2 p=$3 n=$4 x=$5
  local f=$OUT/$b-$p-$n-${x:-dspark}; f=${f// /}
  local PR="$PS"; [ "$p" = 4k ] && PR="$P4"
  HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$(cname $card) bash "$S/run.sh" /work/.scratch/bin/intarsia-$b generate -m $M --backend hip \
    --max-seq 8192 $x --ignore-eos --profile-every 0 --max-tokens "$n" -p "$PR" > "$f.out" 2> "$f.err"
  local rc=$? md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 card$card $b $p $n ${x:-dspark} $md5 rc $rc | $(cat "$f.out" "$f.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) [$(now)]"
}
prime() {  # card binary serve-args (leading NAME=value = env)
  local card=$1 b=$2; shift 2
  local cn; cn=$(cname $card)
  local i0; i0=$(date +%H%M%S)
  echo "[stage] card$card prime $b: $* $(now)"
  docker rm -f "$cn" >/dev/null 2>&1
  local envs=""; while [[ "${1:-}" =~ ^[A-Z_][A-Z0-9_]*= ]]; do envs="$envs $1"; shift; done
  HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$cn nohup bash "$S/run.sh" bash -c "HOME=/work/.scratch/home exec env $envs \
    /work/.scratch/bin/intarsia-$b serve -m $M --backend hip --host 0.0.0.0 --port 8133 --model-id ink $*" \
    > "$OUT/prime-$card-$b-$i0.log" 2>&1 < /dev/null &
  local up=0 i ip
  for i in $(seq 1 170); do
    ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$cn" 2>/dev/null)
    if [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8133/health" > /dev/null 2>&1; then up=$i; break; fi
    sleep 1
  done
  echo "[stage] card$card prime $b up after ${up}s (0 = not up) $(now)"
  grep -aE 'admission|admitted|packed prefill|continuous batching|Error|error' "$OUT/prime-$card-$b-$i0.log" | head -6 | sed "s/^/[card$card] /"
  docker rm -f "$cn" >/dev/null 2>&1
}
card0() {
  prime 0 $BR INTARSIA_INK_PACK=1 --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  for p in short 4k; do for n in 64 256; do gen 0 $MA $p $n "--spec off"; gen 0 $BR $p $n "--spec off"; done; done
  echo "[stage] card0 done $(now)"
}
card1() {
  prime 1 $MA --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  for p in short 4k; do for n in 64 256; do gen 1 $MA $p $n ""; gen 1 $BR $p $n ""; done; done
  echo "[stage] card1 done $(now)"
}
card0 &
card1 &
wait
echo "[stage] done $(now)"
