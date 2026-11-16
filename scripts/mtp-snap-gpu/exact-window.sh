#!/usr/bin/env bash
# Lane mtp-snap-gpu: exactness of PR #395 on HIP (35B, --spec mtp, macro prefill 8192). Driven FROM THE MAC under one fleetq job that takes radiance down:
#   fleetq run --host tatooine --takes-down radiance --expect 40 --mem 30 --quiet gpu,cpu --label "intarsiaml mtp-snap-gpu: exactness" -- scripts/mtp-snap-gpu/exact-window.sh
# ARMS  "<bin>:<scenario>:<rule>" (bin pr|main, scenario c2048|c4096 as in exact.py, rule default|exact = INTARSIA_PREFIX_REUSE unset|exact, macro0 = INTARSIA_PREFILL_MACRO=0): a fresh server, request A
#       (32 tokens), request B (128 tokens), the server goes away.
# COLDS "<bin>:<scenario>": a fresh server that only ever sees B (the cold reference; the reuse rule cannot matter with nothing held).
# c2048 serves with --prefill-chunk 2048 (HIP's recipe chunk is 4096; at 4096 a 2.1k shared prefix has no chunk end to resume at), c4096 with the recipe's chunk.
set -uo pipefail
ARMS=${ARMS:-"pr:c2048:default pr:c2048:exact main:c2048:default"}
COLDS=${COLDS:-"pr:c2048 main:c2048"}
CAP=${CAP:-2400}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-11-16-mtp-snap/raw/gpu/exact-$(date +%H%M)}
. "$HERE/win.sh"
PY="uv run --project /opt/code/deepeval python"
PROMPTS=$REPO/docs/results/2026-11-16-mtp-snap/raw/gpu/prompts.json
URL=http://tatooine:$PORT

send() { # send <A|B> <scenario> <max_tokens> <outfile>
  $PY "$HERE/exact.py" send --url "$URL" --prompts "$PROMPTS" --scenario "$2" --which "$1" --max-tokens "$3" --out "$4" 2>&1 | tee -a "$OUT/steps.log"
}
setup() { # setup <bin> <scenario> <rule>   (sets EXTRA / SENV)
  EXTRA=""; SENV=""
  [[ $2 == c2048 ]] && EXTRA="--prefill-chunk 2048"
  [[ $3 == exact ]] && SENV="INTARSIA_PREFIX_REUSE=exact"
  [[ $3 == macro0 ]] && SENV="INTARSIA_PREFILL_MACRO=0"   # control: macro prefill off (the plain chunk-by-chunk schedule)
}
grab() { # grab <tag>: the plan the server resolved, and the lines that show the macro pass and the chunk
  curl -s -m 10 "$URL/v1/intarsia/plan" > "$OUT/plan-$1.json" 2>&1 < /dev/null
  $SSH "docker logs $CNAME 2>&1 | grep -aiE 'macro|prefill|spec|mtp|chunk' | head -20" < /dev/null > "$OUT/serve-$1.head.log" 2>&1
}

for arm in $ARMS; do
  IFS=: read -r bin sc rule <<<"$arm"
  if (( $(left) < 420 )); then say "SKIPPED warm $arm (time left $(left)s)"; continue; fi
  setup "$bin" "$sc" "$rule"; tag="$bin-$sc-$rule-warm"
  serve "$bin" "$tag" || { drop; continue; }
  grab "$tag"
  send A "$sc" 32 "$OUT/$tag-A.json"
  send B "$sc" 128 "$OUT/$tag-B.json"
  drop
done
for c in $COLDS; do
  IFS=: read -r bin sc <<<"$c"
  if (( $(left) < 300 )); then say "SKIPPED cold $c (time left $(left)s)"; continue; fi
  setup "$bin" "$sc" default; tag="$bin-$sc-cold"
  serve "$bin" "$tag" || { drop; continue; }
  grab "$tag"
  send B "$sc" 128 "$OUT/$tag-B.json"
  drop
done
say "done, time left $(left)s"
