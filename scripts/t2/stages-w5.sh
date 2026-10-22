#!/usr/bin/env bash
# Lane T2 window 5 (ON tatooine, radiance down): re-gate the PR head after merging the newer P1 (#67, now with origin/main
# f9087322) and H1 (#59) heads. Greedy ids must equal windows 1-2 (27B tp=1 / tp=2 short and 4k: 2086b6a4 / ef88d237; 35B
# and 8B tp=1 / tp=2 short), graphs + fusion included; then the HIP crate's GPU tests (main added mxfp4_decode.rs).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/t2/lib.sh"
W2=$HOME/t2/out/w2
ids head 0 27-tp1-short $M27 short
ids head 0 27-tp1-4k $M27 4k
ids head 0,1 27-tp2g-short $M27 short $TP2
E="INTARSIA_TP_FUSION=1" ids head 0,1 27-tp2gf-4k $M27 4k $TP2
ids head 0 35-tp1-short $M35 short
ids head 0,1 35-tp2-short $M35 short $TP2
ids head 0 8-tp1-short $M8 short
ids head 0,1 8-tp2-short $M8 short $TP2
cp "$W2"/ids-*-head.ids "$O/"
same ids-27-tp1-short-head.ids ids-27-tp1-short.ids "27B tp=1 short vs w2"
same ids-27-tp1-4k-head.ids ids-27-tp1-4k.ids "27B tp=1 4k vs w2"
same ids-27-tp1-short-head.ids ids-27-tp2g-short.ids "27B tp=2 graphs short vs w2 tp=1"
same ids-27-tp2gf-4k.ids ids-27-tp1-4k-head.ids "27B tp=2 graphs+fusion 4k vs w2 tp=1"
for x in 35-tp1-short 35-tp2-short 8-tp1-short 8-tp2-short; do same ids-$x-head.ids ids-$x.ids "$x vs w2"; done
log "HIP crate GPU tests"
INTARSIA_REPO=$(T head) INTARSIA_TARGET_VOL=intarsia-cargo-target-t2-head-hiptest HIP_VISIBLE_DEVICES=0,1 timeout 900 \
  "$(T head)/deploy/tatooine/dev.sh" cargo test --release -p intarsia-backend-hip > "$O/hiptest.log" 2>&1 < /dev/null
log "hiptest exit $?: $(grep -E '^test result' "$O/hiptest.log" | awk '{p+=$4; f+=$6} END {print p" passed, "f" failed"}')"
log done
