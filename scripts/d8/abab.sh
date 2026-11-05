#!/usr/bin/env bash
# Lane d8: ABAB served speed of the DFlash default (Qwen3-8B BF16, CUDA, spark1) against `--spec off`, through localeval. Run INSIDE one fleetq job:
#   fleetq run --host spark1 --expect 150 --mem 28 --quiet gpu,cpu --label "d8 abab" -- env W=w1 scripts/d8/abab.sh
# Arms: `off` = `serve --spec off`; `on` = `serve` with NO --spec (the recipe default). A fresh `serve --max-batch 1` per arm. Order ABAB:
# ROUNDS rounds, each off,on then on,off alternately (r1 off on, r2 on off, r3 off on, r4 on off) so slow drift cancels. Per arm, one
# `localeval quality --judge none --max-tokens 256` call per category file of scripts/d8/cats/ (code, math, prose, chat at ~1k and ~4k prompt tokens,
# 4 prompts each); the server's speculation counters are read around each call. Results: docs/results/2026-11-05-dflash-8b-default/raw/$W/.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${W:?W=<window>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
ROUNDS=${ROUNDS:-2}; ORDER=${ORDER:-"off on;on off"}; PORT=${PORT:-8097}; MAXTOK=${MAXTOK:-256}
OUT=$REPO/docs/results/2026-11-05-dflash-8b-default/raw/$W; mkdir -p "$OUT"
H=/home/nvidia/.cache/huggingface/hub
BIN=${BIN:-'~/intarsiaml-d8-target/release/intarsia'}
M8='$(ls -d '$H'/models--Qwen--Qwen3-8B/snapshots/* | head -1)'
URL=http://spark-4e77:$PORT/v1
PIDF='~/intarsiaml-d8-out/serve.pid'; LOGD='~/intarsiaml-d8-out'
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
stop() { ssh spark1 "if [ -f $PIDF ]; then p=\$(cat $PIDF); kill \$p 2>/dev/null; for _ in \$(seq 1 60); do kill -0 \$p 2>/dev/null || break; sleep 1; done; kill -9 \$p 2>/dev/null; rm -f $PIDF; fi" < /dev/null; }
trap stop EXIT INT TERM
metric() { curl -s "http://spark-4e77:$PORT/metrics" | grep -E "^intarsia_spec_(steps|proposed_tokens|accepted_tokens)_total" | tr '\n' ' '; }
ssh spark1 "mkdir -p $LOGD" < /dev/null
r=0
IFS=';' read -r -a SEQS <<< "$ORDER"
for rr in $(seq 1 "$ROUNDS"); do
  order=${SEQS[$(( (rr - 1) % ${#SEQS[@]} ))]}
  for arm in $order; do
    tag="$arm-r$rr"
    case $arm in off) SPEC="--spec off" ;; on) SPEC="" ;; esac
    say "== $tag start"
    ssh spark1 "nohup env ${SERVE_ENV:-} $BIN serve --backend cuda --model $M8 --model-id Qwen3-8B --port $PORT --host 0.0.0.0 --max-batch 1 --max-seq 8192 --mem-budget 24 $SPEC > $LOGD/serve-$W-$tag.log 2>&1 < /dev/null & echo \$! > $PIDF" < /dev/null
    up=0; for _ in $(seq 150); do curl -sf "http://spark-4e77:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
    if [ $up != 1 ]; then say "server $tag did not come up"; ssh spark1 "tail -5 $LOGD/serve-$W-$tag.log" < /dev/null; stop; continue; fi
    ssh spark1 "grep -a -E 'spec dflash|--spec' $LOGD/serve-$W-$tag.log | head -3" < /dev/null | tee -a "$OUT/steps.log"
    for f in "$HERE"/cats/*.jsonl; do
      c=$(basename "$f" .jsonl)
      m0=$(metric)
      ( cd /opt/code/deepeval && uv run localeval quality --url "$URL" --model Qwen3-8B --thinking off --label "d8-$W-$tag-$c" --recipe "intarsia serve --backend cuda --model Qwen3-8B --max-batch 1 --max-seq 8192 $SPEC (lane d8)" \
          --dataset "$f" --judge none --max-tokens "$MAXTOK" --nonce --ignore-busy --no-unim ) > "$OUT/q-$tag-$c.log" 2>&1
      rc=$?
      m1=$(metric)
      echo "$tag $c" >> "$OUT/spec-metrics.txt"; echo "$m0" >> "$OUT/spec-metrics.txt"; echo "$m1" >> "$OUT/spec-metrics.txt"
      say "round $rr arm $arm $c: localeval exit $rc $(grep -aoE 'results/[^ ]+' "$OUT/q-$tag-$c.log" | tail -1)"
    done
    ssh spark1 "cp $LOGD/serve-$W-$tag.log /dev/null" < /dev/null
    scp -q "spark1:intarsiaml-d8-out/serve-$W-$tag.log" "$OUT/" < /dev/null 2>/dev/null
    stop
  done
done
say "== abab done"
