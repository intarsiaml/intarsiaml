#!/usr/bin/env bash
# Lane tp2 window W=w8 (ON tatooine, radiance down, called by window.sh). Trees: D = batch build, B = base (origin/main
# 2c2d994), mut-<name> = mutants (mutants-build.sh). Ordered by priority; every part checks the time left.
#  1. CRASH REPRO (release blocker, coordinator 2026-10-25): w1's first tp=2 accuracy run died once (HSA_STATUS_ERROR_EXCEPTION
#     0x1016, exit 139, right after "this build:", i.e. at load / first launch) on the default config of D. It was the first
#     tp=2 process after a parallel tp=1 prelude (35B on GPU 0, 8B on GPU 1). Needs 0 of N>=12. Three arms, same command each:
#       cold : 6x (the w1 prelude, then at once the tp=2 accuracy run)   [the condition the crash happened in]
#       warm : 24x the tp=2 accuracy run back to back, default config of D
#       v0   : 12x the same with INTARSIA_TP_VOCAB=0 (replicated head)   [is it the new gather?]
#       main : 12x the same on B (tp=2 as main has it)                   [is it older than this lane?]
#  2. correctness mutants of the tp=2 graph / batching / vocab-head logic (+ the batching step-protocol mutants).
#  3. gate check 3 (external reference): KL against the vLLM golden for tp=1 (reference), tp=2 default, tp=2 with a deliberate
#     partial-scale bug (0.3 %, 1 %, 0.1 %) and two equally valid tp=1 variants (no fusion; chunk 233) as controls for the per-prompt
#     noise. scripts/tp2-hip/tp-gate.py reads the acc-ext-*.out files.
#  4. gate check 1 (direct): the HIP GPU test tests/tp_partial_sum.rs (derived fp32 reduction-order bound; mutants), then the rest of
#     the HIP crate tests and `kernels test --quick`.
#  5. prefix reuse at tp=2; vocab/arnorm dumps (bitwise); the gate vs tp=1 on the long golden (reported, not gating).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-batch}; B=${B:-base}; NW=${NW:-24}; NC=${NC:-6}; NV=${NV:-12}; NB=${NB:-12}
CAP=${CAP:-2400}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { if (( $(left) > $1 )); then return 0; fi; log "SKIPPED (time left $(left)s < $1): $2"; return 1; }
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8"
log "window start, CAP ${CAP}s"

# ---- 1. crash repro
one() { # one <tree> <label> <i>: the w1 command; returns the exit code
  local tree=$1 l=$2 i=$3
  run "$tree" 0,1 300 "repro-$l-$i" accuracy --backend hip --model $M35 --golden $SG --mode both --mem-budget 29 --max-seq 17408 \
    --write-self-golden /work/tp2-out/repro-$l-$i.safetensors $TP2 --ref-noise $NS --write-verdict /work/tp2-out/verdict-repro-$l-$i.json > /dev/null
  local rc=$?
  if [[ $l == warm && $i == 1 && $rc == 0 ]]; then mv "$(T "$tree")/tp2-out/repro-$l-$i.safetensors" "$O/" 2>/dev/null; else rm -f "$(T "$tree")/tp2-out/repro-$l-$i.safetensors"; fi
  # keep only the err of failures
  if [[ $rc == 0 ]]; then rm -f "$O/repro-$l-$i.err" "$O/repro-$l-$i.out"; fi
  return $rc
}
prelude() { # the parallel tp=1 work w1 did just before its crashing run
  ( acc $D 0 pre35-$1 $M35 $SG > /dev/null 2>&1 ) &
  ( gen $D 1 pre8-$1 $M8 short > /dev/null 2>&1 ) &
  wait
  rm -f "$O"/acc-pre35-$1.* "$O"/gen-pre8-$1.* 2>/dev/null
}
arm() { # arm <tree> <label> <count> <cold?>
  local tree=$1 l=$2 n=$3 cold=$4 bad=0 i rcs=""
  for i in $(seq 1 "$n"); do
    need 150 "repro $l $i" || break
    [[ $cold == cold ]] && prelude $i
    one "$tree" "$l" "$i"; local rc=$?
    rcs="$rcs$rc "
    [[ $rc != 0 ]] && bad=$((bad + 1))
  done
  log "REPRO $l: $bad of $(wc -w <<<"$rcs") runs failed (exit codes: $rcs)"
}
arm "$D" cold "$NC" cold
arm "$D" warm "$NW" no
need 120 "run-to-run dump" && [[ -f $O/repro-warm-1.safetensors ]] && \
  self $D 0,1 tp2-r2-vs-r1 repro-warm-1.safetensors $TP2 $FLOOR
E="INTARSIA_TP_VOCAB=0" arm "$D" v0 "$NV" no
arm "$B" main "$NB" no
log "repro done, time left $(left)s"

# ---- 2. mutants
if need 300 "mutants"; then
  gen $D 0,1 mut-ref $M35 short $TP2
  for m in no-peer-upload drop-captured-allreduce peer-ignores-flag tp-fold-outproj batch-peer-no-table vocab-no-gather; do
    need 90 "mutant $m" || continue
    [[ -x $(T mut-$m)/tp2-intarsia ]] || { log "MUTANT $m: no binary"; continue; }
    gen mut-$m 0,1 mut-$m $M35 short $TP2
    if cmp -s "$O/gen-mut-ref.ids" "$O/gen-mut-$m.ids"; then v="ids SAME as the clean tp=2 run (SURVIVED)"; else v="ids DIFFER (killed)"; fi
    log "MUTANT $m: $v; $(grep -ahE 'tensor-parallel rank 1: decode graph|^Error|trap|Aborted|core dumped|HSA_STATUS' "$O/gen-mut-$m.err" | head -3 | tr '\n' '|')"
  done
fi
if need 420 "batch mutants"; then
  mkdir -p "$O/bm"; ONLY=mut D=$D B=$B bash "$HOME/tp2/stages-batch.sh" "$O/bm" < /dev/null
  log "batch mutants: $(grep -ah 'mtest' "$O/bm/steps.log" | cut -c1-300 | tr '\n' '|')"
fi

# ---- 3. check 3: distance to the external reference (vLLM golden), mean and per prompt
if need 700 "check 3 runs"; then
  acc $D 0 ext-tp1 $M35 $SG
  acc $D 0,1 ext-tp2 $M35 $SG $TP2
  E="INTARSIA_TP_TEST_BUG=scale:1.003" acc $D 0,1 ext-tp2-bug0.3 $M35 $SG $TP2
  E="INTARSIA_TP_TEST_BUG=scale:1.01" acc $D 0,1 ext-tp2-bug1 $M35 $SG $TP2
  E="INTARSIA_TP_TEST_BUG=scale:1.001" acc $D 0,1 ext-tp2-bug0.1 $M35 $SG $TP2
  E="INTARSIA_NO_FUSION=1" acc $D 0 ext-tp1-nofusion $M35 $SG
  acc $D 0 ext-tp1-chunk233 $M35 $SG --prefill-chunk 233
  log "check 3 outputs: $(ls "$O"/acc-ext-*.out | wc -l) tables (read with scripts/tp2-hip/tp-gate.py)"
fi

# ---- 4. check 1: the direct partial-sum test, then the HIP crate tests
ht() { # ht <tag> <cargo test args...>
  local tag=$1; shift
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest timeout 420 "$(T $D)/deploy/tatooine/dev.sh" \
    cargo test --release -p intarsia-backend-hip --no-fail-fast "$@" > "$O/hiptest-$tag.log" 2>&1 < /dev/null
  log "hiptest $tag exit $?: $(grep -aE '^test result' "$O/hiptest-$tag.log" | awk '{p+=$4; f+=$6; i+=$8} END {print p " passed, " f " failed, " i " ignored"}') $(grep -aE 'FAILED|panicked|HSA_STATUS' "$O/hiptest-$tag.log" | head -4 | tr '\n' '|')"
}
need 300 "tp_partial_sum" && ht tp-partial-sum --test tp_partial_sum -- --nocapture --test-threads=1
need 300 "all hip tests" && ht all
run $D 0 420 kernels-test-quick kernels test --backend hip --device 0 --model $M35 --quick
log "kernels test --quick: $(grep -aiE 'cases|fail' "$O/kernels-test-quick.out" | tail -2 | tr '\n' '|')"

# ---- 5. reuse, dumps, long gate
if need 500 "reuse"; then
  run $D 0,1 400 reuse-tp2 accuracy --backend hip --model $M35 --golden $LG --mode reuse --mem-budget 29 --max-seq 17408 $TP2
  E="INTARSIA_TP_FUSION=0" run $D 0,1 400 reuse-tp2-fusion0 accuracy --backend hip --model $M35 --golden $LG --mode reuse --mem-budget 29 --max-seq 17408 $TP2
fi
if need 200 "vocab dump"; then
  E="INTARSIA_TP_VOCAB=0" acc $D 0,1 vocab0-35b-short $M35 $SG $TP2
  self $D 0,1 vocab1-vs-vocab0-short acc-vocab0-35b-short.safetensors $TP2 --pre-change ${PRECHANGE:-2c2d994} $FLOOR
fi
if need 200 "arnorm dump"; then
  acc $D 0,1 arn0-35b-short $M35 $SG $TP2
  E="INTARSIA_TP_ARNORM=1" self $D 0,1 arn1-vs-arn0-short acc-arn0-35b-short.safetensors $TP2 $FLOOR
  log "evidence arnorm dump: $(grep -ah 'fused all-reduce+rmsnorm' "$O/self-arn1-vs-arn0-short.err" | tr '\n' '|')"
fi
if need 600 "long gate"; then
  run $D 0 500 acc-tp1-35b-long accuracy --backend hip --model $M35 --golden $LG --mode both --mem-budget 29 --max-seq 17408 \
    --write-self-golden /work/tp2-out/acc-tp1-35b-long.safetensors
  mv "$(T $D)/tp2-out/acc-tp1-35b-long.safetensors" "$O/" 2>/dev/null
  cp "$O/acc-tp1-35b-long.safetensors" "$(T $D)/tp2-out/"
  run $D 0,1 500 self-tp2-vs-tp1-long accuracy --backend hip --model $M35 --mem-budget 29 --max-seq 17408 --self-golden /work/tp2-out/acc-tp1-35b-long.safetensors \
    --mode both --allow-same-code --allow-config-mismatch $TP2 --self-vs-ref-noise --ref-noise $NL --ref-noise-factor 1.0 --write-verdict /work/tp2-out/verdict-gate-long.json
fi
log "done, time left $(left)s"
