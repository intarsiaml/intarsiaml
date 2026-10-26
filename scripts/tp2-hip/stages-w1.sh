#!/usr/bin/env bash
# Lane tp2 window w1 (ON tatooine, radiance down, called by window.sh): the tp>1 gate "vs tp1" (user decision 2026-10-25) and
# the tp=1 no-op. Trees: D = the stack build (stage 1 + vocab-head + ar-norm; sync.sh stack HEAD), B = origin/main 2c2d994
# (code-identical base of the lane; the tree `base`).
#  A (GPU 0: 35B, GPU 1: 8B, in parallel) tp=1 no-op: plan show json, 256 greedy ids short and 4k, 35B accuracy dump (the tp1
#     self-golden for the gate below). D must equal B bit for bit.
#  B (both GPUs) tp=2 on D (HIP defaults: graphs + fusion + vocab-parallel head):
#     1. engine-vs-golden (vLLM short golden, reference noise) as before;
#     2. THE GATE: the tp2 dump against the tp1 dump, `--self-golden --self-vs-ref-noise` with the recipe's tp_ref_noise_factor;
#     3. its sensitivity: INTARSIA_TP_TEST_BUG=scale:F on rank 1 (a deliberate partial-scale bug) must FAIL the same gate;
#     4. run to run: a second tp2 dump and 256 ids short / 4k twice must be identical (floor budgets, 1e-13 residual class).
#  C the three opt-outs against the default at tp=2: ids with INTARSIA_TP_VOCAB=0 / INTARSIA_TP_ARNORM=1 / INTARSIA_TP_FUSION=0.
# Every check logs SAME / DIFF or the gate line in steps.log; the dumps are fetched by window.sh for the offline cmpdump.py.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-stack}; B=${B:-base}
CAP=${CAP:-2400}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { if (( $(left) > $1 )); then return 0; fi; log "SKIPPED (time left $(left)s < $1): $2"; return 1; }
PV="--pre-change ${PRECHANGE:-2c2d994}"
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8"
GATE="--self-vs-ref-noise --ref-noise $NS"
for t in $D $B; do echo "$t: $(cat "$(T $t)/tp2-intarsia.stamp" 2>/dev/null || ls -la "$(T $t)/tp2-intarsia")"; done > "$O/versions.txt"
log "window start, CAP ${CAP}s"

# ---- A. tp=1 no-op, D against B
track35() {
  for t in $B $D; do
    plan $t 0 tp1-$t-35b $M35
    acc $t 0 tp1-$t-35b-short $M35 $SG
    gen $t 0 tp1-$t-35b-short $M35 short
    gen $t 0 tp1-$t-35b-4k $M35 4k
  done
}
track8() {
  for t in $B $D; do
    plan $t 1 tp1-$t-8b $M8
    gen $t 1 tp1-$t-8b-short $M8 short
    gen $t 1 tp1-$t-8b-4k $M8 4k
  done
}
track35 & p1=$!
track8 & p2=$!
wait $p1 $p2
same plan-tp1-$B-35b.out plan-tp1-$D-35b.out "35B tp=1 plan show json, D vs B"
same plan-tp1-$B-8b.out plan-tp1-$D-8b.out "8B tp=1 plan show json, D vs B"
for k in short 4k; do
  same gen-tp1-$B-35b-$k.ids gen-tp1-$D-35b-$k.ids "35B tp=1 256 ids $k, D vs B"
  same gen-tp1-$B-8b-$k.ids gen-tp1-$D-8b-$k.ids "8B tp=1 256 ids $k, D vs B"
done
self $D 0 tp1-D-vs-B-short acc-tp1-$B-35b-short.safetensors $PV $FLOOR   # tp=1 accuracy dump: bitwise (floor budgets)
log "A done, time left $(left)s"

# ---- B. tp=2
need 600 "B1 engine-vs-golden" && acc $D 0,1 tp2-35b-short $M35 $SG $TP2 --ref-noise $NS --write-verdict /work/tp2-out/verdict-tp2-engine-short.json
# the gate (the factor comes from recipes/qwen3_5_moe.toml [accuracy] tp_ref_noise_factor)
need 400 "B2 gate" && self $D 0,1 tp2-vs-tp1-short acc-tp1-$B-35b-short.safetensors $TP2 $PV $GATE --write-verdict /work/tp2-out/verdict-gate-short.json
# the same numbers at the old factor (continuity with H1's 1.66x)
need 400 "B2 gate at factor 1.0" && self $D 0,1 tp2-vs-tp1-short-f1 acc-tp1-$B-35b-short.safetensors $TP2 $PV $GATE --ref-noise-factor 1.0
# sensitivity: deliberate partial-scale bugs on rank 1 must FAIL the gate
for f in 1.01 1.003; do
  need 400 "B3 bug scale:$f" && E="INTARSIA_TP_TEST_BUG=scale:$f" self $D 0,1 tp2-bug$f-vs-tp1-short acc-tp1-$B-35b-short.safetensors $TP2 $PV $GATE
done
# run to run
need 300 "B4 ids" && for k in short 4k; do
  gen $D 0,1 tp2-r1-35b-$k $M35 $k $TP2
  gen $D 0,1 tp2-r2-35b-$k $M35 $k $TP2
  same gen-tp2-r1-35b-$k.ids gen-tp2-r2-35b-$k.ids "35B tp=2 256 ids $k: run 1 vs run 2"
done
need 400 "B4 dump" && { acc $D 0,1 tp2-35b-short-r2 $M35 $SG $TP2; self $D 0,1 tp2-r2-vs-r1-short acc-tp2-35b-short.safetensors $TP2 $FLOOR; }
log "B done, time left $(left)s"

# ---- C. opt-outs against the default
if need 600 "C opt-outs"; then
  E="INTARSIA_TP_VOCAB=0" gen $D 0,1 tp2-vocab0-35b-short $M35 short $TP2
  same gen-tp2-r1-35b-short.ids gen-tp2-vocab0-35b-short.ids "35B tp=2 ids short: INTARSIA_TP_VOCAB=0 vs default"
  E="INTARSIA_TP_ARNORM=1" gen $D 0,1 tp2-arnorm1-35b-short $M35 short $TP2
  same gen-tp2-r1-35b-short.ids gen-tp2-arnorm1-35b-short.ids "35B tp=2 ids short: INTARSIA_TP_ARNORM=1 vs default"
  E="INTARSIA_TP_FUSION=0" gen $D 0,1 tp2-fusion0-35b-short $M35 short $TP2
  same gen-tp2-r1-35b-short.ids gen-tp2-fusion0-35b-short.ids "35B tp=2 ids short: INTARSIA_TP_FUSION=0 vs default"
fi
mv "$(T $D)"/tp2-out/verdict-*.json "$O/" 2>/dev/null
log "done"
