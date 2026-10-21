#!/usr/bin/env bash
# Lane H1 window 1 (ON tatooine, radiance down, called by window.sh). Trees: base = origin/main 0294b77, dev = lane-h1.
#  A (in parallel): GPU 0 the 35B tp=1 no-op (plan, accuracy dumps, 256 ids short / 4k) and the tp=1 calibration runs
#     (equally correct reorderings: SIMT attention, a small prefill chunk, another GEMV tiling; a kernel trace);
#     GPU 1 the 8B tp=1 no-op.
#  B (both GPUs): tp=2. Short-set first (accuracy outranks speed): base eager reference, dev graphs (engine-vs-golden
#     gate with the reference noise, and change-vs-base against base tp=1 with --self-vs-ref-noise), the KL diagnostics
#     on base (host communicator, bf16 partials, kernel trace), then ids (base eager / dev graphs / dev --no-graphs /
#     dev graphs+fusion), the fusion dump, the 8B tp=2, and the long set last.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/h1/lib.sh"
SIMT=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash
PV="--pre-change 0294b77"
for t in base dev; do echo "$t: $(cat "$(T $t)/h1-intarsia.stamp" 2>/dev/null || ls -la "$(T $t)/h1-intarsia")"; done > "$O/versions.txt"

track35() {
  for t in base dev; do
    plan $t 0 $t-35b $M35
    acc $t 0 $t-35b-short $M35 $SG
    gen $t 0 $t-35b-short $M35 short
    gen $t 0 $t-35b-4k $M35 4k
  done
  acc base 0 base-35b-long $M35 $LG
  # calibration on base tp=1, short set: what does an equally correct reordering cost on HIP?
  E="INTARSIA_HIP_DISABLE=$SIMT" acc base 0 cal-simt-short $M35 $SG
  acc base 0 cal-chunk128-short $M35 $SG --prefill-chunk 128
  E="INTARSIA_HIP_GEMV=4,8" acc base 0 cal-gemv48-short $M35 $SG
  E="INTARSIA_HIP_TRACE=1" acc base 0 trace-tp1-short $M35 $SG
}
track8() {
  for t in base dev; do
    plan $t 1 $t-8b $M8
    acc $t 1 $t-8b-short $M8 $G/qwen3-8b.safetensors
    gen $t 1 $t-8b-short $M8 short
    gen $t 1 $t-8b-4k $M8 4k
  done
}
# 0. capture diagnostics: the HIP comm tests (prebuilt, volume ...-hiptest) in parallel (the harness default) and one
#    thread at a time, with the HIP runtime's error log; then a dev tp=2 smoke with graphs (rank 1's counters on exit)
hiptest() { # hiptest <tag> [cargo test args]
  local tag=$1; shift
  INTARSIA_REPO=$(T dev) INTARSIA_TARGET_VOL=intarsia-cargo-target-h1-dev-hiptest timeout 300 "$(T dev)/deploy/tatooine/dev.sh" \
    env AMD_LOG_LEVEL=1 cargo test --release -p intarsia-backend-hip "$@" > "$O/hiptest-$tag.log" 2>&1 < /dev/null
  log "hiptest $tag exit $?: $(grep -aE '^test result|FAILED|panicked' "$O/hiptest-$tag.log" | head -6 | tr '\n' '|')"
}
hiptest comm-par --test comm
hiptest comm-ser --test comm -- --test-threads=1
E="AMD_LOG_LEVEL=1" gen dev 0,1 smoke-dev-35b-tp2g-short $M35 short $TP2

log "A: tp=1 tracks start"
track35 & p35=$!
track8 & p8=$!
wait $p35 $p8
log "A: tp=1 tracks done"
for m in 35b 8b; do
  same plan-base-$m.out plan-dev-$m.out "tp=1 plan $m"
  for k in short 4k; do same gen-base-$m-$k.ids gen-dev-$m-$k.ids "tp=1 ids $m $k"; done
done

log "B: tp=2 start"
acc base 0,1 base-35b-tp2-short $M35 $SG $TP2 --ref-noise $NS
# KL diagnostics on base: is it the communicator, the partial precision, or the kernels' selection?
acc base 0,1 diag-tp2-commhost-short $M35 $SG $TP2 --comm host
E="INTARSIA_TP_PARTIAL=bf16" acc base 0,1 diag-tp2-partbf16-short $M35 $SG $TP2
E="INTARSIA_HIP_TRACE=1" acc base 0,1 trace-tp2-short $M35 $SG $TP2
E="INTARSIA_HIP_DISABLE=$SIMT" acc base 0,1 diag-tp2-simt-short $M35 $SG $TP2
acc dev 0,1 dev-35b-tp2g-short $M35 $SG $TP2 --ref-noise $NS --write-verdict /work/h1-out/verdict-dev-tp2g-short.json
self dev 0,1 dev-tp2g-vs-base-tp1-short acc-base-35b-short.safetensors $TP2 $PV --self-vs-ref-noise --ref-noise $NS
self dev 0,1 dev-tp2g-vs-base-tp2e-short acc-base-35b-tp2-short.safetensors $TP2 $PV --self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8
# ids: graphs vs eager vs --no-graphs vs fusion
for k in short 4k; do
  gen base 0,1 base-35b-tp2-$k $M35 $k $TP2
  gen dev 0,1 dev-35b-tp2g-$k $M35 $k $TP2
  gen dev 0,1 dev-35b-tp2ng-$k $M35 $k $TP2 --no-graphs
  E="INTARSIA_TP_FUSION=1" gen dev 0,1 dev-35b-tp2gf-$k $M35 $k $TP2
  same gen-base-35b-tp2-$k.ids gen-dev-35b-tp2g-$k.ids "tp=2 ids $k: base eager vs dev graphs"
  same gen-dev-35b-tp2ng-$k.ids gen-dev-35b-tp2g-$k.ids "tp=2 ids $k: dev --no-graphs vs dev graphs"
  same gen-dev-35b-tp2gf-$k.ids gen-dev-35b-tp2g-$k.ids "tp=2 ids $k: dev graphs+fusion vs dev graphs"
done
E="INTARSIA_TP_FUSION=1" acc dev 0,1 dev-35b-tp2gf-short $M35 $SG $TP2
# the 8B tp=2 path (graphs now on by default there too)
gen base 0,1 base-8b-tp2-short $M8 short $TP2
gen dev 0,1 dev-8b-tp2g-short $M8 short $TP2
gen dev 0,1 dev-8b-tp2ng-short $M8 short $TP2 --no-graphs
same gen-base-8b-tp2-short.ids gen-dev-8b-tp2g-short.ids "8b tp=2 ids: base eager vs dev graphs"
same gen-dev-8b-tp2ng-short.ids gen-dev-8b-tp2g-short.ids "8b tp=2 ids: dev --no-graphs vs dev graphs"
acc base 0,1 base-8b-tp2-short $M8 $G/qwen3-8b.safetensors $TP2
acc dev 0,1 dev-8b-tp2g-short $M8 $G/qwen3-8b.safetensors $TP2
# long set last
acc base 0,1 base-35b-tp2-long $M35 $LG $TP2 --ref-noise $NL
acc dev 0,1 dev-35b-tp2g-long $M35 $LG $TP2 --ref-noise $NL --write-verdict /work/h1-out/verdict-dev-tp2g-long.json
self dev 0,1 dev-tp2g-vs-base-tp1-long acc-base-35b-long.safetensors $TP2 $PV --self-vs-ref-noise --ref-noise $NL
mv "$(T dev)"/h1-out/verdict-*.json "$O/" 2>/dev/null
log "done"
