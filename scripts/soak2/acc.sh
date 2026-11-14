#!/usr/bin/env bash
# soak2 lane: GSM8K n=200 after a prefix-store restore, MTP on (the shipping default), thinking off, greedy. usage: acc.sh [arms="W O"]
#   fleetq run --host spark1 --quiet gpu,cpu --mem 55 --expect 40 --label "soak2: gsm8k restore" -- bash scripts/soak2/acc.sh
# scripts/soak/gsm8k_prefix.py: system prompt + 32 worked few-shot turns shared by every request (6.9k tokens), the test question last.
#   W  `serve --max-batch 2 --prefix-store on` (default 8 GiB), concurrency 1: the first request is cold, the rest restore the shared prefix
#   O  `serve --max-batch 2` (no store), concurrency 1: every request prefills in full; same MTP default
# Both arms at concurrency 1 so every decode drafts (a sequence only drafts while it decodes alone) and only the store differs.
set -uo pipefail
ARMS=${1:-"W O"}
WLOG=window-acc
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
[[ -f $OUT/gsm8k-inputs.json ]] || { echo "missing $OUT/gsm8k-inputs.json"; exit 1; }
prep_box
gp() { # gp <arm> <concurrency>
  echo "== $1 gsm8k-prefix conc=$2 $(date +%T)"
  ( cd /opt/code/deepeval && uv run python "$SOAKS/gsm8k_prefix.py" --url "$URL" --model $MODEL --inputs "$OUT/gsm8k-inputs.json" --arm "$1" \
      --out "$OUT" --concurrency "$2" --shots 32 --limit 200 --max-tokens 512 ) > "$OUT/$1-gp.out" 2>&1
  tail -2 "$OUT/$1-gp.out" | cut -c1-900 | tee -a "$OUT/${WLOG}.log"
  scrape "$OUT/serve-$1.metrics.after-$1"
  python3 "$SC/metric_diff.py" "$OUT/serve-$1.metrics.before" "$OUT/serve-$1.metrics.after-$1" | tee "$OUT/$1.storediff.json" | tee -a "$OUT/${WLOG}.log"
}
for arm in $ARMS; do
  case $arm in
    W) args="--max-batch 2 --prefix-store on";;
    O) args="--max-batch 2";;
    *) log "unknown arm $arm"; continue;;
  esac
  if start_serve $arm $args; then gp $arm 1; finish_serve $arm; fi
done
