#!/usr/bin/env bash
# Lane W4A8 ids gate, ON tatooine, INSIDE a GPU window that holds both cards (radiance down; w3.sh calls it): Qwen3.8-27B MXFP4 at
# tp2 with --mxfp4-act fp8: for each prompt of scripts/cmtp/ids-prompts (short-0..5, 4k-0..5, 16k-0..5) greedy ids (256, --ignore-eos)
# with --spec off and with the shipped MTP default (no --spec flag); every `auto` arm must equal `off`.
#   ids.sh [prompt names...]   default: all 18.   ACT=fp8 (default) | none | dynamic   O=<out dir>
set -uo pipefail
R=$HOME/intarsiaml-w4a8; V=intarsia-cargo-w4a8; D=$R/deploy/tatooine/dev.sh
O=${O:-$HOME/intarsiaml-w4a8-logs/ids}; mkdir -p "$O"
ACT=${ACT:-fp8}
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
H=/data/models/radiance/hf-cache/hub
P=$R/scripts/cmtp/ids-prompts
names=("$@"); [ ${#names[@]} -gt 0 ] || names=(short-0 short-1 short-2 short-3 short-4 short-5 4k-0 4k-1 4k-2 4k-3 4k-4 4k-5 16k-0 16k-1 16k-2 16k-3 16k-4 16k-5)
fails=0
for n in "${names[@]}"; do
  case $n in short*) MS=4096 ;; 4k*) MS=8192 ;; *) MS=20480 ;; esac
  for arm in off auto; do
    case $arm in off) A="--spec off" ;; auto) A="" ;; esac
    INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V HIP_VISIBLE_DEVICES=0,1 INTARSIA_MODELS=$H $D /target/release/intarsia generate -m $M27 --backend hip --tp 2 --devices 0,1 \
      --mem-budget 29 --max-seq $MS $A --mxfp4-act $ACT --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
      -p "$(cat $P/$n.txt)" --max-tokens ${NTOK:-256} < /dev/null > "$O/gen-$n-$arm.out" 2> "$O/gen-$n-$arm.err"
    grep -a '^ids:' "$O/gen-$n-$arm.err" > "$O/gen-$n-$arm.ids"
  done
  st=$(grep -a '^spec:' "$O/gen-$n-auto.err" | tail -1 | cut -c1-110)
  if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-auto.ids" && [ -s "$O/gen-$n-off.ids" ] && [ -n "$st" ]; then
    echo "PASS ids $n $(tr ',' '\n' < "$O/gen-$n-off.ids" | wc -l) ids identical | $st"
  else echo "FAIL ids $n differ/empty/no spec line | $st"; fails=$((fails + 1)); tail -3 "$O/gen-$n-auto.err" | cut -c1-200; fi
done
echo "ids fails=$fails"
