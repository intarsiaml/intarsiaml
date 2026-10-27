#!/usr/bin/env bash
# Lane H27P ids gate, ON tatooine INSIDE a GPU window that holds both cards (win.sh calls it): Qwen3.8-27B MXFP4 at tp2 (W4A8, the
# recipe default), for each prompt of scripts/cmtp/ids-prompts (short-0..5, 4k-0..5, 16k-0..5) greedy ids (NTOK=256, --ignore-eos) from
#   base  = ~/intarsiaml-h27p/base-intarsia (origin/main d877564c8, chunk 256) --spec off
#   dev   = ~/intarsiaml-h27p/h27p-intarsia (this branch, chunk 2048 by recipe) --spec off
#   devc  = dev with --prefill-chunk 256 (the flag restores the old schedule on the new binary) --spec off   [only with CHK=1]
#   auto  = dev, the shipped MTP default (no --spec flag)
# PASS iff base == dev == auto (and devc when CHK=1) id by id, and the spec line exists.
#   ids.sh [prompt names...]   default: all 18.   O=<out dir>  EXTRA="<extra generate args for dev/auto/devc>"
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
O=${O:-$HOME/intarsiaml-h27p-logs/ids}; mkdir -p "$O"
P=$R/scripts/cmtp/ids-prompts
names=("$@"); [ ${#names[@]} -gt 0 ] || names=(short-0 short-1 short-2 short-3 short-4 short-5 4k-0 4k-1 4k-2 4k-3 4k-4 4k-5 16k-0 16k-1 16k-2 16k-3 16k-4 16k-5)
fails=0
run1() { # run1 <binary name> <tag> <ms> <prompt> <extra args...>
  local bin=$1 tag=$2 ms=$3 n=$4; shift 4
  gpu 230 $([ "$tag" != base ] && echo ${DEVENV:-}) /work/$bin generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq $ms \
    --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids -p "$(cat $P/$n.txt)" --max-tokens ${NTOK:-256} "$@" \
    > "$O/gen-$n-$tag.out" 2> "$O/gen-$n-$tag.err"
  grep -a '^ids:' "$O/gen-$n-$tag.err" > "$O/gen-$n-$tag.ids"
}
for n in "${names[@]}"; do
  case $n in short*) MS=4096 ;; 4k*) MS=8192 ;; *) MS=20480 ;; esac
  { [ "${REUSE_BASE:-0}" = 1 ] && [ -s "$O/gen-$n-base.ids" ]; } || run1 base-intarsia base $MS $n --spec off
  run1 h27p-intarsia dev $MS $n --spec off ${EXTRA:-}
  run1 h27p-intarsia auto $MS $n ${EXTRA:-}
  [ "${CHK:-0}" = 1 ] && run1 h27p-intarsia devc $MS $n --spec off --prefill-chunk 256 ${EXTRA:-}
  st=$(grep -a '^spec:' "$O/gen-$n-auto.err" | tail -1 | cut -c1-110)
  ok=1
  for t in dev auto; do cmp -s "$O/gen-$n-base.ids" "$O/gen-$n-$t.ids" || ok=0; done
  [ "${CHK:-0}" = 1 ] && { cmp -s "$O/gen-$n-base.ids" "$O/gen-$n-devc.ids" || ok=0; }
  [ -s "$O/gen-$n-base.ids" ] && [ -n "$st" ] || ok=0
  chunk=$(grep -a 'prefill chunk' "$O/gen-$n-dev.err" | head -1 | cut -c1-60)
  if [ $ok = 1 ]; then echo "PASS ids $n $(tr ',' '\n' < "$O/gen-$n-base.ids" | wc -l) ids identical base=dev=auto | $chunk | $st"
  else echo "FAIL ids $n | $chunk | $st"; fails=$((fails + 1)); for t in base dev auto; do echo "  $t: $(md5sum < "$O/gen-$n-$t.ids" | cut -c1-8)"; done; tail -2 "$O/gen-$n-dev.err" | cut -c1-200; fi
done
echo "ids fails=$fails"
