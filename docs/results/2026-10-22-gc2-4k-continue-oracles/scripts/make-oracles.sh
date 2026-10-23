#!/usr/bin/env bash
# gc2: continue oracles (fx5c oracle-relative gate) for the 35B 4k golden's long_docs_4k, splits 2000 (chunk <= 1024) and 3072 (chunk 2048).
# Same recipe as scripts/fx5c (seq-chain.sh): the engine's cold greedy stream is frozen (here Metal on endor, chunk 1024 / 2048; fx5c used CUDA),
# then the f64 CPU oracle (accuracy --backend cpu --mode prefill, chunk 256, top-64) scores one prefill over turn2 + stream.
#   fleetq run --host endor --mem 42 --expect 75 --label "gc2: 4k-golden continue oracles" -- make-oracles.sh OUTDIR
set -u
: "${FLEETQ_JOB:?run under fleetq}"
OUT=${1:?outdir}; mkdir -p "$OUT"; O=$(cd "$OUT" && pwd)
cd "$(git rev-parse --show-toplevel)"
BIN=${BIN:-/opt/code/wt-gc-main-target/release/intarsia}      # origin/main a92267b0, metal; has --reuse-dump / --reuse-seq
M=/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4
G4=$PWD/golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors
FX=$PWD/scripts/fx5c
STAGE=$O/oracle-files; mkdir -p $STAGE
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/job.log"; }
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-23 05:00" +%s)
for CS in 1024:2000 2048:3072; do
  C=${CS%%:*}; S=${CS##*:}; D=$O/s$S; mkdir -p $D/dump
  [ $(date +%s) -lt $DEADLINE ] || { log "deadline: skip s$S"; continue; }
  log "s$S: Metal cold stream, chunk $C"
  $BIN accuracy --model $M --backend metal --golden $G4 --mode reuse --max-seq 8192 --mem-budget 30 --allow-dirty --reuse-diag \
    --reuse-prompt long_docs_4k --prefill-chunk $C --reuse-dump $D/dump < /dev/null > $D/dump.log 2>&1
  log "s$S dump exit $?"; cp $D/dump/seq.json $D/seq.json 2>/dev/null || { log "s$S: no seq.json"; continue; }
  python3 -c "import json;d=json.load(open('$D/seq.json'));print('split',d['split'],'chunk',d.get('chunk'),'m',d['m'],'row0',d['row0'],'turn2',len(d['turn2']))" | tee -a $O/job.log
  rm -f $D/dump/*.f32
  uv run --project tools python $FX/make_oracle_input.py $D/seq.json $G4 $D/oracle-input.safetensors 2>&1 | tail -1 | tee -a $O/job.log
  t0=$(date +%s)
  /usr/bin/time -l $BIN accuracy --model $M --backend cpu --golden $D/oracle-input.safetensors --mode prefill --prefill-chunk 256 --mem-budget 40 \
    --max-seq 8192 --allow-dirty --write-self-golden $D/oracle.dump.safetensors < /dev/null > $D/oracle.log 2>&1
  log "s$S oracle exit $? $(( $(date +%s) - t0 )) s; $(grep -a 'maximum resident' $D/oracle.log | head -1)"
  uv run --project tools python $FX/make_continue_oracle.py $D/seq.json $D/oracle.dump.safetensors $G4 long_docs_4k $STAGE \
    --machine "endor (Apple M5 Max, CPU backend)" --run "gc2 $(basename $O); frozen stream: Metal cold engine, chunk $C, 4k golden" 2>&1 | tail -3 | tee -a $O/job.log
  rm -f $D/oracle-input.safetensors
done
# the 4k golden's files must not collide with the long golden's (same model / prompt / split): add a suffix, fix the sidecar
for f in $STAGE/*-s*.continue-oracle.safetensors; do
  [ -e "$f" ] || continue; b=${f%.continue-oracle.safetensors}
  case "$b" in *-g4k) continue;; esac
  mv "$f" "$b-g4k.continue-oracle.safetensors"; mv "$f.json" "$b-g4k.continue-oracle.safetensors.json"
  sed -i '' "s#$(basename $f)#$(basename $b)-g4k.continue-oracle.safetensors#" "$b-g4k.continue-oracle.safetensors.json"
done
ls -l $STAGE | tee -a $O/job.log; log done
