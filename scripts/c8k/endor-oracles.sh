#!/usr/bin/env bash
# Lane c8k, endor side: continue oracles (fx5c / lt recipe) for prompts the prefix-reuse gate cannot yet judge by oracle:
#   Metal cold greedy stream (skipped when <work>/<golden>-<prompt>-c<chunk>/seq.json exists from metal-dump.sh) ->
#   make_oracle_input.py -> f64 CPU oracle (`accuracy --backend cpu --mode prefill --prefill-chunk 256`, top-64) ->
#   make_continue_oracle.py -> golden/oracle (4k golden files get the fx5c `-g4k` suffix).
# Inside a fleetq slot on endor (--load cpu --mem 45).
# Usage: endor-oracles.sh <workdir> <outdir> <golden:prompt:chunk:maxseq> ...      golden: long | 4k
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
W=$1; OUTD=$2; shift 2; mkdir -p "$W" "$OUTD"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN=${BIN:-/opt/code/wt-gc-main-target/release/intarsia}   # metal + cpu backends, main a92267b0 (gc2's oracle binary)
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
FX=$REPO/scripts/fx5c
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$W/chain.log"; }
for spec in "$@"; do
  IFS=: read -r G pr chunk maxseq <<< "$spec"
  GOLD=$REPO/golden/qwen36-35b-a3b-nvfp4-vllm-$G-top20.safetensors
  D=$W/$G-$pr-c$chunk; mkdir -p "$D/dump"
  if [[ ! -s $D/seq.json ]]; then
    log "$spec: Metal cold stream"
    "$BIN" accuracy --model $M --backend metal --golden "$GOLD" --mode reuse --max-seq $maxseq --mem-budget 40 --allow-dirty --no-reuse-oracle \
      --reuse-diag --reuse-prompt $pr --prefill-chunk $chunk --reuse-dump "$D/dump" < /dev/null > "$D/dump.log" 2>&1
    log "$spec: dump exit $?"; cp "$D/dump/seq.json" "$D/seq.json" 2>/dev/null || { log "$spec: no seq.json"; continue; }
  fi
  python3 -c "import json;d=json.load(open('$D/seq.json'));print('split',d['split'],'chunk',d.get('chunk'),'m',d['m'],'row0',d['row0'],'turn2',len(d['turn2']))" | tee -a "$W/chain.log"
  rm -f "$D"/dump/*.f32
  uv run --project "$REPO/tools" python $FX/make_oracle_input.py "$D/seq.json" "$GOLD" "$D/oracle-input.safetensors" 2>&1 | tail -1 | tee -a "$W/chain.log"
  t0=$(date +%s)
  /usr/bin/time -l "$BIN" accuracy --model $M --backend cpu --golden "$D/oracle-input.safetensors" --mode prefill --prefill-chunk 256 --mem-budget 40 \
    --max-seq $maxseq --allow-dirty --write-self-golden "$D/oracle.dump.safetensors" < /dev/null > "$D/oracle.log" 2>&1
  log "$spec: oracle exit $? $(( $(date +%s) - t0 )) s; $(grep -a 'maximum resident' $D/oracle.log | head -1)"
  STAGE=$D/stage; rm -rf "$STAGE"; mkdir -p "$STAGE"
  uv run --project "$REPO/tools" python $FX/make_continue_oracle.py "$D/seq.json" "$D/oracle.dump.safetensors" "$GOLD" $pr "$STAGE" \
    --machine "endor (Apple M5 Max, CPU backend, $(basename $BIN))" --run "c8k $(date +%F), frozen stream: Metal cold engine, chunk $chunk, golden $G" 2>&1 | tail -3 | tee -a "$W/chain.log"
  rm -f "$D/oracle-input.safetensors"
  for f in "$STAGE"/*.continue-oracle.safetensors; do
    [ -e "$f" ] || continue; b=${f%.continue-oracle.safetensors}
    if [[ $G == 4k ]]; then
      mv "$f" "$b-g4k.continue-oracle.safetensors"; mv "$f.json" "$b-g4k.continue-oracle.safetensors.json"
      sed -i '' "s#$(basename $f)#$(basename $b)-g4k.continue-oracle.safetensors#" "$b-g4k.continue-oracle.safetensors.json"
    fi
  done
  cp "$STAGE"/* "$OUTD"/; ls -l "$STAGE" | tee -a "$W/chain.log"
done
log done
