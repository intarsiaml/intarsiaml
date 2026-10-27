#!/usr/bin/env bash
# Lane tp2 window W=27b (ON tatooine, radiance down, called by window.sh): Qwen3.8-27B MXFP4 (W4A4) at tp=2 with the lane's
# defaults (graphs, fusion, vocab-parallel head; ARNORM and batching as opt-ins) against tp=1. T2 showed tp=2 bitwise tp=1
# for this model (every MXFP4 product and its f32 sum is exact), so the gate is BITWISE: the tp2 dump against the tp1 dump
# with the floor budgets, plus the 256 ids. Trees: D = batch (the stack + batched tp2), B = base (tp=1 reference build).
#   fleetq run ... -- env W=27b CAP=2400 XENV="D=batch B=base" scripts/tp2-hip/window.sh
#  1. plans (fused all-reduces, vocab-parallel head at tp=2, tp=1 plan D vs B)
#  2. tp=1: accuracy dump (27B golden, P1's), 256 ids short/4k (GPU 0), D and B (tp=1 no-op)
#  3. tp=2 default: ids short/4k (== tp=1), accuracy dump, `self` against the tp=1 dump (floor budgets = bitwise) and the
#     --self-vs-ref-noise form
#  4. tp=2 opt-ins/outs against the default: INTARSIA_TP_VOCAB=0, INTARSIA_TP_ARNORM=1 (fused counters printed), INTARSIA_TP_FUSION=0
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-batch}; B=${B:-base}
CAP=${CAP:-2400}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { if (( $(left) > $1 )); then return 0; fi; log "SKIPPED (time left $(left)s < $1): $2"; return 1; }
HUB=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
G27=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-top20.safetensors
N27=/work/golden/ref_noise/qwen3.8-27b-mxfp4-hfquark-top20.safetensors.json
export INTARSIA_MODELS=$HUB
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8"
for t in $D $B; do cp -rn "$HOME/intarsiaml-p1/p1-golden" "$(T $t)/" 2>/dev/null; echo "$t: $(cat "$(T $t)/tp2-intarsia.stamp" 2>/dev/null || ls -la "$(T $t)/tp2-intarsia")"; done > "$O/versions.txt"
# the 27B accuracy run: golden, reference-noise sidecar, dump
acc27() { # acc27 <tree> <gpus> <tag> [args]
  local tree=$1 gpus=$2 tag=$3; shift 3
  run "$tree" "$gpus" 600 "acc-$tag" accuracy --backend hip --model $M27 --mem-budget 29 --golden $G27 --mode both \
    --mxfp4-act dynamic --ref-noise $N27 --write-self-golden /work/tp2-out/acc-$tag.safetensors --allow-unstamped "$@"
  mv "$(T "$tree")/tp2-out/acc-$tag.safetensors" "$O/" 2>/dev/null
}
self27() { # self27 <tree> <gpus> <tag> <dump in $O> [args]
  local tree=$1 gpus=$2 tag=$3 d=$4; shift 4
  cp "$O/$d" "$(T "$tree")/tp2-out/$d"
  run "$tree" "$gpus" 600 "self-$tag" accuracy --backend hip --model $M27 --mem-budget 29 --mode both --self-golden /work/tp2-out/$d \
    --allow-same-code --allow-config-mismatch --allow-unstamped --pre-change ${PRECHANGE:-2c2d994} "$@"
}
gen27() { gen "$@"; }
log "window start, CAP ${CAP}s"

# 1/2. plans and tp=1 (D and B), GPU 0 only
for t in $B $D; do
  plan $t 0 tp1-$t-27b $M27
  gen $t 0 tp1-$t-27b-short $M27 short
  gen $t 0 tp1-$t-27b-4k $M27 4k
done
same plan-tp1-$B-27b.out plan-tp1-$D-27b.out "27B tp=1 plan show json, D vs B"
for k in short 4k; do same gen-tp1-$B-27b-$k.ids gen-tp1-$D-27b-$k.ids "27B tp=1 256 ids $k, D vs B"; done
need 500 "tp1 dump" && acc27 $D 0 tp1-27b
log "tp1 done, time left $(left)s"

# 3. tp=2 default
plan $D 0,1 tp2-27b $M27 $TP2
log "tp2 plan: $(grep -aoE '"lm_head": "[^"]*"|"all_reduce_calls_per_decode_step": [0-9]+|"all_reduce_fused_rmsnorm_per_decode_step": [0-9]+' "$O/plan-tp2-27b.out" | tr '\n' '|')"
for k in short 4k; do
  gen $D 0,1 tp2-27b-$k $M27 $k $TP2
  same gen-tp1-$D-27b-$k.ids gen-tp2-27b-$k.ids "27B 256 ids $k: tp=2 default vs tp=1 (want SAME: W4A4 is exact)"
done
if need 600 "tp2 dump + gates"; then
  acc27 $D 0,1 tp2-27b $TP2
  self27 $D 0,1 tp2-vs-tp1 acc-tp1-27b.safetensors $TP2 $FLOOR
  self27 $D 0,1 tp2-vs-tp1-noise acc-tp1-27b.safetensors $TP2 --self-vs-ref-noise --ref-noise $N27
fi

# 4. switches against the default
for sw in "INTARSIA_TP_VOCAB=0" "INTARSIA_TP_ARNORM=1" "INTARSIA_TP_FUSION=0"; do
  need 300 "ids $sw" || continue
  E="$sw" gen $D 0,1 tp2-${sw%%=*}-27b-short $M27 short $TP2
  same gen-tp2-27b-short.ids gen-tp2-${sw%%=*}-27b-short.ids "27B tp=2 ids short: $sw vs default"
  log "evidence $sw: $(grep -ah 'fused all-reduce+rmsnorm' "$O/gen-tp2-${sw%%=*}-27b-short.err" | tr '\n' '|')"
done
log "done, time left $(left)s"
