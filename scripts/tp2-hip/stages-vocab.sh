#!/usr/bin/env bash
# Lane tp2, window stage W=vocab (ON tatooine, radiance down, called by window.sh): the vocab-parallel lm_head
# (LoadOptions::tp_vocab, INTARSIA_TP_VOCAB; branch lane-tp2/vocab-head). Trees: D = the vocab-head build
# (scripts/tp2-hip/sync.sh vocab <rev> + build.sh, HIPTEST=1), B = origin/main (the lane's existing `base` tree).
#   D=vocab B=base bash stages-vocab.sh <out dir>        QUICK=1 skips the 4k, 8B and speed stages
# Every check logs SAME / DIFF (see lib.sh `same`) or a gate line; "A/B" means the same binary with INTARSIA_TP_VOCAB=1 / 0
# (the CLI default at HIP tp=2 is on; the explicit values keep the stage independent of the default).
#  1. device tests of the new collective and the split: comm.rs (all_gather rank-major, in place and disjoint, > 1 slot,
#     captured in a graph between all-reduces, 200 mixed rounds) and vocab_split.rs (the half-size head calls are bitwise the
#     two halves of the full-size call, m = 1..16, bf16 and NVFP4, tied view);
#  2. tp=2 35B NVFP4: A/B 256 greedy ids short (graphs), --no-graphs with the split head vs graphs without it, 4k; plan shows
#     the split; accuracy dumps A/B (short set) and `self` with the floor budgets = BITWISE (the head is all that changed);
#     a control: INTARSIA_TP_NO_COMM=1 (no gather) must DIFFER, which shows the ids check can fail;
#  3. tp=1 unchanged: plan show and 256 ids (short, 4k) of D vs B at tp=1 (GPU 0) and the 8B (GPU 1);
#  4. tp=2 8B (dense bf16 head): A/B ids; speed: decode tok/s of 4k generations, alternating A/B x3.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-vocab}; B=${B:-base}
QUICK=${QUICK:-0}
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8"
A_ON="INTARSIA_TP_VOCAB=1"; A_OFF="INTARSIA_TP_VOCAB=0"
for t in $D $B; do echo "$t: $(cat "$(T $t)/tp2-intarsia.stamp" 2>/dev/null || ls -la "$(T $t)/tp2-intarsia")"; done > "$O/versions.txt"
tok_s() { grep -ahoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' "$1" | tail -1; }

# 1. device tests (prebuilt by build.sh HIPTEST=1 in the volume ...-$D-hiptest: no rebuild here)
hiptest() { # hiptest <tag> <cargo test args...>
  local tag=$1; shift
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest timeout 600 "$(T $D)/deploy/tatooine/dev.sh" \
    cargo test --release -p intarsia-backend-hip "$@" -- --test-threads=1 --nocapture > "$O/hiptest-$tag.log" 2>&1 < /dev/null
  log "hiptest $tag exit $?: $(grep -aE '^test result|FAILED|panicked|SKIPPED' "$O/hiptest-$tag.log" | head -6 | tr '\n' '|')"
}
hiptest comm --test comm
hiptest vocab-split --test vocab_split
hiptest kernels-unit --lib kernels

# 2. tp=2 35B: the split head against the replicated one, ids first (accuracy outranks speed)
E="$A_OFF" gen $D 0,1 tp2-off-35b-short $M35 short $TP2
E="$A_ON" gen $D 0,1 tp2-on-35b-short $M35 short $TP2
same gen-tp2-off-35b-short.ids gen-tp2-on-35b-short.ids "35B tp=2 256 greedy ids short, split head vs replicated (graphs)"
E="$A_ON" gen $D 0,1 tp2-on-nograph-35b-short $M35 short $TP2 --no-graphs
same gen-tp2-off-35b-short.ids gen-tp2-on-nograph-35b-short.ids "35B tp=2 256 ids short, split head eager vs replicated graphs"
log "rank 1 graph counters (split head): $(grep -ah 'tensor-parallel rank 1: decode graph' "$O/gen-tp2-on-35b-short.err" | tr '\n' '|')"
E="$A_ON INTARSIA_TP_NO_COMM=1" gen $D 0,1 tp2-control-nocomm-35b-short $M35 short $TP2
same gen-tp2-off-35b-short.ids gen-tp2-control-nocomm-35b-short.ids "CONTROL (no collectives, must DIFFER)"
plan $D 0,1 tp2-on-35b $M35 $TP2
log "plan tp=2 split head: $(grep -aoE '"lm_head": "[^"]*"|"all_reduce_calls_per_decode_step": [0-9]+' "$O/plan-tp2-on-35b.out" | tr '\n' '|')"
E="$A_OFF" plan $D 0,1 tp2-off-35b $M35 $TP2
log "plan tp=2 replicated head: $(grep -aoE '"lm_head": "[^"]*"|"all_reduce_calls_per_decode_step": [0-9]+' "$O/plan-tp2-off-35b.out" | tr '\n' '|')"
E="$A_OFF" acc $D 0,1 tp2-off-35b-short $M35 $SG $TP2
E="$A_ON" acc $D 0,1 tp2-on-35b-short $M35 $SG $TP2
if command -v uv > /dev/null; then
  (cd "$HOME/tp2" && uv run --with numpy --with safetensors python cmpdump.py "$O/acc-tp2-off-35b-short.safetensors" "$O/acc-tp2-on-35b-short.safetensors") > "$O/cmpdump-short.txt" 2>&1
  log "cmpdump short (A/B accuracy dumps): $(tail -1 "$O/cmpdump-short.txt")"
fi
# change-vs-base with the floor budgets: the split head must be bitwise the replicated one (and `self` shows the table)
E="$A_ON" self $D 0,1 on-vs-off-short acc-tp2-off-35b-short.safetensors $TP2 $FLOOR
if [[ $QUICK != 1 ]]; then
  E="$A_OFF" gen $D 0,1 tp2-off-35b-4k $M35 4k $TP2
  E="$A_ON" gen $D 0,1 tp2-on-35b-4k $M35 4k $TP2
  same gen-tp2-off-35b-4k.ids gen-tp2-on-35b-4k.ids "35B tp=2 256 greedy ids at 4k context, split head vs replicated"
fi

# 3. tp=1 unchanged (GPU 0: 35B; GPU 1: 8B), the new build against origin/main
for t in $D $B; do
  plan $t 0 tp1-$t-35b $M35
  gen $t 0 tp1-$t-35b-short $M35 short
done
same plan-tp1-$D-35b.out plan-tp1-$B-35b.out "35B tp=1 plan show json, D vs B"
same gen-tp1-$D-35b-short.ids gen-tp1-$B-35b-short.ids "35B tp=1 256 ids short, D vs B"
if [[ $QUICK != 1 ]]; then
  for t in $D $B; do gen $t 0 tp1-$t-35b-4k $M35 4k; gen $t 1 tp1-$t-8b-short $M8 short; done
  same gen-tp1-$D-35b-4k.ids gen-tp1-$B-35b-4k.ids "35B tp=1 256 ids 4k, D vs B"
  same gen-tp1-$D-8b-short.ids gen-tp1-$B-8b-short.ids "8B tp=1 256 ids short, D vs B"

# 4. the dense bf16 head (8B), then speed
  E="$A_OFF" gen $D 0,1 tp2-off-8b-short $M8 short $TP2
  E="$A_ON" gen $D 0,1 tp2-on-8b-short $M8 short $TP2
  same gen-tp2-off-8b-short.ids gen-tp2-on-8b-short.ids "8B tp=2 256 ids short (dense bf16 head), split vs replicated"
  for i in 1 2 3; do
    E="$A_OFF" gen $D 0,1 tp2-speed-off-$i-35b-4k $M35 4k $TP2
    E="$A_ON" gen $D 0,1 tp2-speed-on-$i-35b-4k $M35 4k $TP2
    log "speed $i: replicated $(tok_s "$O/gen-tp2-speed-off-$i-35b-4k.err") | split $(tok_s "$O/gen-tp2-speed-on-$i-35b-4k.err")"
  done
  for i in 1 2; do
    E="$A_OFF" gen $D 0,1 tp2-speed-off-$i-8b-4k $M8 4k $TP2
    E="$A_ON" gen $D 0,1 tp2-speed-on-$i-8b-4k $M8 4k $TP2
    log "speed 8B $i: replicated $(tok_s "$O/gen-tp2-speed-off-$i-8b-4k.err") | split $(tok_s "$O/gen-tp2-speed-on-$i-8b-4k.err")"
  done
fi
log "done"
