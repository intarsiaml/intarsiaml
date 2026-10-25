#!/usr/bin/env bash
# Lane tp2 window W=w16 (ON tatooine, radiance down, window.sh with SNAP=batch and STAGES=w16): the two open findings of w15.
#  A. 27B batched == solo at tp=2 panics at N=4 (step 0, row 0, ALL 248320 logits differ, first 2.473 solo vs 2.269 batch); N=2,3 passed.
#     1. rows_exact: MXFP4 rows kernel == the one-row GEMV at the 27B's tp=2 shard shapes, M = 1..16 (names the shape/M);
#     2. the batched test at tp=2 with one switch off at a time (vocab-parallel head, fusion, batched graphs): which switch moves it;
#     3. the same test at tp=1 on this tree (does #145's 27B pass reproduce here).
#  B. 27B tp=2 against tp=1 at W4A16 differs by 1.26e-3 mean KL (max 1.33 at chat_multiturn position 12, where tp=1 is already 1.285 from the
#     reference): tp=2 on its own against the golden, twice (determinism), with dumps for a per-position three-way comparison offline.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-batch}; PC=${PC:?PC=<head sha of the build>}
CAP=${CAP:-1700}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { if (( $(left) > $1 )); then return 0; fi; log "SKIPPED (time left $(left)s < $1): $2"; return 1; }
vram() { local m=0 v c; for c in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do v=$(cat "$c" 2>/dev/null || echo 0); (( v > m )) && m=$v; done; echo $m; }
clear() { local i; for i in 1 2 3 4 5 6; do (( $(vram) < 1073741824 )) && return 0; sleep 10; done
  log "CONTENDED before '$1': $(( $(vram) / 1048576 )) MiB in use on a card: NO VERDICT"; return 1; }
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
G27=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
export INTARSIA_MODELS=/data/models/radiance/hf-cache/hub
log "window start, CAP ${CAP}s, pre-change $PC, VRAM in use $(( $(vram) / 1048576 )) MiB"
mkdir -p "$(T $D)/tp2-out"
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8 --pre-change $PC"
# A1. the rows kernel at the shard shapes
if clear "rows_exact"; then
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest HIP_VISIBLE_DEVICES=0 timeout 400 \
    "$(T $D)/deploy/tatooine/dev.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_exact mxfp4_rows_are_bitwise_the_one_row_gemv -- --test-threads=1 --nocapture \
    > "$O/hiptest-rows-mxfp4.log" 2>&1 < /dev/null
  log "rows_exact mxfp4 exit $?: $(grep -aE '^test result|panicked|differ|MISMATCH|mismatch|SKIPPED' "$O/hiptest-rows-mxfp4.log" | head -6 | cut -c1-240 | tr '\n' '|')"
fi
# B. 27B accuracy: tp=1 and tp=2 (twice) against the W4A16 golden, dumps kept
if need 400 "27B accuracy" && clear "27B accuracy"; then
  run $D 0 600 acc-27b-tp1 accuracy --backend hip --model $M27 --mxfp4-act none --mem-budget 29 --golden $G27 --mode both --write-self-golden /work/tp2-out/acc-27b-tp1.safetensors
  mv "$(T $D)/tp2-out/acc-27b-tp1.safetensors" "$O/" 2>/dev/null
  for r in a b; do
    run $D 0,1 600 acc-27b-tp2-$r accuracy --backend hip --model $M27 --mxfp4-act none --mem-budget 29 --golden $G27 --mode both \
      --write-self-golden /work/tp2-out/acc-27b-tp2-$r.safetensors $TP2
    mv "$(T $D)/tp2-out/acc-27b-tp2-$r.safetensors" "$O/" 2>/dev/null
  done
  cp "$O/acc-27b-tp2-a.safetensors" "$(T $D)/tp2-out/"
  run $D 0,1 600 self-27b-tp2-b-vs-a accuracy --backend hip --model $M27 --mxfp4-act none --mem-budget 29 --mode both \
    --self-golden /work/tp2-out/acc-27b-tp2-a.safetensors --allow-same-code --allow-config-mismatch $TP2 $FLOOR
fi
# A2. the batched test at tp=2, one switch at a time (it stops at its first mismatch: ~1 min each); A3. at tp=1 (full, ~12 min)
bt() { # bt <tag> <gpus> <timeout> <env...>
  local tag=$1 gpus=$2 to=$3; shift 3
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest HIP_VISIBLE_DEVICES=$gpus timeout "$to" \
    "$(T $D)/deploy/tatooine/dev.sh" env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR="$M27" "$@" \
    cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture --exact real_35b_batched_decode_is_bitwise_the_solo_step \
    > "$O/bt-$tag.log" 2>&1 < /dev/null
  log "bt $tag exit $? [$*]: $(grep -aE '^test result|batch [0-9]+ step|pairs bitwise' "$O/bt-$tag.log" | head -3 | cut -c1-260 | tr '\n' '|')"
}
for v in "default:INTARSIA_BATCH_TEST_TP=2" "vocab0:INTARSIA_BATCH_TEST_TP=2 INTARSIA_TP_VOCAB=0" "fusion0:INTARSIA_BATCH_TEST_TP=2 INTARSIA_TP_FUSION=0" \
         "graphs0:INTARSIA_BATCH_TEST_TP=2 INTARSIA_BATCH_GRAPHS=0"; do
  need 200 "bt ${v%%:*}" && clear "bt ${v%%:*}" && bt "tp2-${v%%:*}" 0,1 150 ${v#*:}
done
if need 950 "27B batched at tp1" && clear "27B batched tp1"; then bt tp1 0 900 INTARSIA_BATCH_TEST_TP=1; fi
log "done, time left $(left)s"
