#!/usr/bin/env bash
# Lane H27P tail sweep, ON tatooine INSIDE a GPU window (win.sh): prompts of ~4.1k tokens cut from ids-prompts/4k-5 by K*4 characters, K = 0..44
# (the token count falls ~1 per cut, so the length mod 256 sweeps 33 -> 0 -> 255 -> ~245: every tail of 1..16 rows and its neighbours), greedy
# ids (NTOK=48) from base-intarsia (chunk 256) and h27p-intarsia (the recipe: chunk 2048 + peel 16) at tp2, --spec off. PASS iff all equal.
#   tails.sh [K list]   O=<out dir>  NOPEEL=1 runs dev with INTARSIA_PREFILL_PEEL=0 (shows the peel is what keeps the tail identical)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
O=${O:-$HOME/intarsiaml-h27p-logs/tails}; mkdir -p "$O"
SRC=$R/scripts/cmtp/ids-prompts/4k-5.txt; L=$(wc -c < "$SRC")
ks=("$@"); [ ${#ks[@]} -gt 0 ] || ks=($(seq 0 44))
pass=0; fail=0
for k in "${ks[@]}"; do
  head -c $((L - 4 * k)) "$SRC" > "$O/p$k.txt"
  for t in base dev; do
    bin=base-intarsia; envs=""; [ $t = dev ] && { bin=h27p-intarsia; [ "${NOPEEL:-0}" = 1 ] && envs="env INTARSIA_PREFILL_PEEL=0"; }
    gpu 230 $envs /work/$bin generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 8192 --spec off \
      --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids -p "$(cat $O/p$k.txt)" --max-tokens ${NTOK:-48} \
      > "$O/gen-$k-$t.out" 2> "$O/gen-$k-$t.err"
    grep -a '^ids:' "$O/gen-$k-$t.err" > "$O/gen-$k-$t.ids"
  done
  n=$(grep -ao "prompt=[0-9]*" "$O/gen-$k-dev.err" | head -1 | cut -d= -f2)
  if cmp -s "$O/gen-$k-base.ids" "$O/gen-$k-dev.ids" && [ -s "$O/gen-$k-base.ids" ]; then r=PASS; pass=$((pass + 1)); else r=FAIL; fail=$((fail + 1)); fi
  echo "$r tail k=$k prompt=$n mod256=$((n % 256)) peel=$(grep -ac 'peel' "$O/gen-$k-dev.err")"
  rm -f "$O/p$k.txt"
done
echo "tails pass=$pass fail=$fail"
