#!/usr/bin/env bash
# Lane tp2, window stage W=vocab (ON tatooine, radiance down, called by window.sh): the vocab-parallel lm_head
# (LoadOptions::tp_vocab, INTARSIA_TP_VOCAB; branch lane-tp2/vocab-head). Trees: D = the vocab-head build
# (sync.sh vocab <rev>, build.sh TREES=vocab=<stamp> HIPTEST=1), B = origin/main (the lane's existing `base` tree).
#   W=vocab ... window.sh        (D=vocab B=base are the defaults; stages-vocab2.sh is the second window: 4k, 8B, speed)
# "A/B" = the same binary with INTARSIA_TP_VOCAB=1 / 0 (the CLI default at HIP tp=2 is on; the explicit values keep the
# stage independent of the default). Every check logs SAME / DIFF (lib.sh `same`) or a gate line in steps.log. Rough budget,
# model loads dominate (~1-1.5 min per generate): ~30 min, in priority order, so a CAP cut loses the least important last.
#  1. device tests (~5 min): comm.rs (all_gather rank-major, in place and disjoint, > 1 scratch slot, captured in a graph
#     between all-reduces, mixed with eager rounds), vocab_split.rs (the half-size head calls are bitwise the two halves of the
#     full-size call, m = 1..16, bf16 and NVFP4, tied row view), the kernels.rs selection pin;
#  2. tp=2 35B NVFP4 (~7 min): 256 greedy ids short, A/B (graphs), split head --no-graphs vs replicated graphs, and a CONTROL
#     (INTARSIA_TP_NO_COMM=1: no collectives) that must DIFFER, which shows the ids check can fail;
#  3. tp=1 unchanged (~6 min, GPU 0 = 35B plan + ids of D and B, GPU 1 = 8B ids, in parallel): D == B bit for bit;
#  4. tp=2 plan A/B (the split must show `vocab-parallel`, else the stage logs VACUOUS) and the accuracy dumps A/B on the short
#     golden (~12 min): `self` of the split run against the replicated dump with the floor budgets = bitwise.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-vocab}; B=${B:-base}
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8"
A_ON="INTARSIA_TP_VOCAB=1"; A_OFF="INTARSIA_TP_VOCAB=0"
for t in $D $B; do echo "$t: $(cat "$(T $t)/tp2-intarsia.stamp" 2>/dev/null || ls -la "$(T $t)/tp2-intarsia")"; done > "$O/versions.txt"
head_of() { grep -aoE '"lm_head": "[^"]*"|"all_reduce_calls_per_decode_step": [0-9]+' "$1" | tr '\n' '|'; }

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

# 2. tp=2 35B: the split head against the replicated one
E="$A_OFF" gen $D 0,1 tp2-off-35b-short $M35 short $TP2
E="$A_ON" gen $D 0,1 tp2-on-35b-short $M35 short $TP2
same gen-tp2-off-35b-short.ids gen-tp2-on-35b-short.ids "35B tp=2 256 greedy ids short, split head vs replicated (graphs)"
log "rank 1 graph counters (split head): $(grep -ah 'tensor-parallel rank 1: decode graph' "$O/gen-tp2-on-35b-short.err" | tr '\n' '|')"
E="$A_ON" gen $D 0,1 tp2-on-nograph-35b-short $M35 short $TP2 --no-graphs
same gen-tp2-off-35b-short.ids gen-tp2-on-nograph-35b-short.ids "35B tp=2 256 ids short, split head eager vs replicated graphs"
E="$A_ON INTARSIA_TP_NO_COMM=1" gen $D 0,1 tp2-control-nocomm-35b-short $M35 short $TP2
same gen-tp2-off-35b-short.ids gen-tp2-control-nocomm-35b-short.ids "CONTROL (split head, no collectives: must DIFFER)"

# 3. tp=1 unchanged: D against B, 35B on GPU 0 and the 8B on GPU 1 in parallel
track35() {
  for t in $D $B; do plan $t 0 tp1-$t-35b $M35; gen $t 0 tp1-$t-35b-short $M35 short; done
}
track8() {
  for t in $D $B; do gen $t 1 tp1-$t-8b-short $M8 short; done
}
track35 & p1=$!
track8 & p2=$!
wait $p1 $p2
same plan-tp1-$D-35b.out plan-tp1-$B-35b.out "35B tp=1 plan show json, D vs B"
if ! cmp -s "$O/plan-tp1-$D-35b.out" "$O/plan-tp1-$B-35b.out"; then log "plan diff: $(diff <(tr ',' '\n' < "$O/plan-tp1-$D-35b.out") <(tr ',' '\n' < "$O/plan-tp1-$B-35b.out") | head -6 | tr '\n' '|')"; fi
same gen-tp1-$D-35b-short.ids gen-tp1-$B-35b-short.ids "35B tp=1 256 ids short, D vs B"
same gen-tp1-$D-8b-short.ids gen-tp1-$B-8b-short.ids "8B tp=1 256 ids short, D vs B"

# 4. plan and accuracy A/B at tp=2
E="$A_ON" plan $D 0,1 tp2-on-35b $M35 $TP2
E="$A_OFF" plan $D 0,1 tp2-off-35b $M35 $TP2
log "plan tp=2 split: $(head_of "$O/plan-tp2-on-35b.out") | replicated: $(head_of "$O/plan-tp2-off-35b.out")"
if grep -aq 'vocab-parallel' "$O/plan-tp2-on-35b.out" && ! grep -aq 'vocab-parallel' "$O/plan-tp2-off-35b.out"; then
  log "split head ACTIVE with INTARSIA_TP_VOCAB=1 and off with =0 (the A/B above is not vacuous)"
else
  log "VACUOUS: the plan does not show the vocab-parallel head with INTARSIA_TP_VOCAB=1 (or shows it with =0): every A/B above compares the replicated head with itself"
fi
E="$A_OFF" acc $D 0,1 tp2-off-35b-short $M35 $SG $TP2
E="$A_ON" acc $D 0,1 tp2-on-35b-short $M35 $SG $TP2
# the split run against the replicated dump with the floor budgets: bitwise or the gate line says why not; the two dumps are
# also fetched by window.sh for the offline `cmpdump.py` comparison
E="$A_ON" self $D 0,1 on-vs-off-short acc-tp2-off-35b-short.safetensors $TP2 $FLOOR
log "done"
