#!/usr/bin/env bash
# Lane tp2 window W=w8 (ON tatooine, radiance down, called by window.sh): the intermittent HSA exception, the mutants, the
# prefix-reuse gate and the HIP crate GPU tests. Trees: D = batch build, B = base (origin/main 2c2d994), mut-<name> = mutants
# (mutants-build.sh).
#  1. CRASH REPRO. w1's first tp=2 accuracy run died once (HSA_STATUS_ERROR_EXCEPTION 0x1016, exit 139) with the default
#     config on D; the same command ran clean afterwards. The same command N times on D, on B (tp=2 as main has it), and on D
#     with INTARSIA_TP_VOCAB=0, each with its exit code. A fault only on D with the vocab-parallel head points at the new gather.
#  2. prefix reuse at tp=2 (the long golden, `--mode reuse`), default config and INTARSIA_TP_FUSION=0.
#  3. the HIP crate GPU tests (all), `kernels test --backend hip --quick`.
#  4. mutants of the graph capture / batching / vocab head: 256 ids short at tp=2 against D's own tp=2 ids; killed = ids differ,
#     rank 1's replay counters differ from rank 0's, a trap or a timeout.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-batch}; B=${B:-base}; N=${N:-8}
CAP=${CAP:-2400}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { if (( $(left) > $1 )); then return 0; fi; log "SKIPPED (time left $(left)s < $1): $2"; return 1; }
log "window start, CAP ${CAP}s"
# 1. the exact w1 B1 command, N times
repro() { # repro <tree> <label> [E]
  local tree=$1 l=$2 bad=0 i
  for i in $(seq 1 "$N"); do
    need 120 "repro $l $i" || break
    run "$tree" 0,1 300 "repro-$l-$i" accuracy --backend hip --model $M35 --golden $SG --mode both --mem-budget 29 --max-seq 17408 \
      --write-self-golden /work/tp2-out/repro-$l-$i.safetensors $TP2 --ref-noise $NS --write-verdict /work/tp2-out/verdict-repro-$l-$i.json > /dev/null
    [[ $? != 0 ]] && bad=$((bad + 1))
    if [[ $l == D && $i == 1 ]]; then mv "$(T "$tree")/tp2-out/repro-$l-$i.safetensors" "$O/" 2>/dev/null; else rm -f "$(T "$tree")/tp2-out/repro-$l-$i.safetensors"; fi
  done
  log "REPRO $l: $bad of $N runs failed"
}
repro "$D" D
# tp=2 run to run: a fresh dump against the first one, floor budgets (bitwise class)
need 120 "run to run dump" && [[ -f $O/repro-D-1.safetensors ]] && self $D 0,1 tp2-r2-vs-r1 repro-D-1.safetensors $TP2 --self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8
E="INTARSIA_TP_VOCAB=0" repro "$D" D-vocab0; repro "$B" B
journal() { dmesg 2>/dev/null | grep -i -E "amdgpu|gpu|fault" | tail -5 | tr '\n' '|'; }
log "dmesg: $(journal)"
# 1b. the gate on the long golden as well (H1: 0.89x / 0.79x, under the noise): tp=1 dump, tp=2 against it, factor 1.0
if need 600 "long gate"; then
  run $D 0 500 acc-tp1-35b-long accuracy --backend hip --model $M35 --golden $LG --mode both --mem-budget 29 --max-seq 17408 \
    --write-self-golden /work/tp2-out/acc-tp1-35b-long.safetensors
  mv "$(T $D)/tp2-out/acc-tp1-35b-long.safetensors" "$O/" 2>/dev/null
  cp "$O/acc-tp1-35b-long.safetensors" "$(T $D)/tp2-out/"
  run $D 0,1 500 self-tp2-vs-tp1-long accuracy --backend hip --model $M35 --mem-budget 29 --max-seq 17408 --self-golden /work/tp2-out/acc-tp1-35b-long.safetensors \
    --mode both --allow-same-code --allow-config-mismatch $TP2 --self-vs-ref-noise --ref-noise $NL --ref-noise-factor 1.0 --write-verdict /work/tp2-out/verdict-gate-long.json
fi
# 2. prefix reuse at tp=2
if need 500 "reuse"; then
  run $D 0,1 400 reuse-tp2 accuracy --backend hip --model $M35 --golden $LG --mode reuse --mem-budget 29 --max-seq 17408 $TP2
  E="INTARSIA_TP_FUSION=0" run $D 0,1 400 reuse-tp2-fusion0 accuracy --backend hip --model $M35 --golden $LG --mode reuse --mem-budget 29 --max-seq 17408 $TP2
fi
# 2b. fused all-reduce + rmsnorm (opt-in): a fresh INTARSIA_TP_ARNORM=1 dump against the default's, floor budgets (bitwise class)
if need 200 "arnorm dump"; then
  acc $D 0,1 arn0-35b-short $M35 $SG $TP2
  E="INTARSIA_TP_ARNORM=1" self $D 0,1 arn1-vs-arn0-short acc-arn0-35b-short.safetensors $TP2 --self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8
  log "evidence arnorm dump: $(grep -ah 'fused all-reduce+rmsnorm' "$O/self-arn1-vs-arn0-short.err" | tr '\n' '|')"
fi
# 3. GPU tests
if need 600 "hip crate tests"; then
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest timeout 540 "$(T $D)/deploy/tatooine/dev.sh" \
    cargo test --release --no-fail-fast -p intarsia-backend-hip > "$O/hiptest-all.log" 2>&1 < /dev/null
  log "hiptest all exit $?: $(grep -aE '^test result' "$O/hiptest-all.log" | awk '{p+=$4; f+=$6} END {print p " passed, " f " failed"}') $(grep -aE 'FAILED|panicked' "$O/hiptest-all.log" | head -4 | tr '\n' '|')"
fi
run $D 0 420 kernels-test-quick kernels test --backend hip --device 0 --model $M35 --quick
log "kernels test --quick: $(grep -aiE 'cases|fail' "$O/kernels-test-quick.out" | tail -2 | tr '\n' '|')"
# 4. mutants
if need 300 "mutants"; then
  gen $D 0,1 mut-ref $M35 short $TP2
  for m in no-peer-upload drop-captured-allreduce peer-ignores-flag tp-fold-outproj batch-peer-no-table vocab-no-gather; do
    need 90 "mutant $m" || continue
    [[ -x $(T mut-$m)/tp2-intarsia ]] || { log "MUTANT $m: no binary"; continue; }
    gen mut-$m 0,1 mut-$m $M35 short $TP2
    if cmp -s "$O/gen-mut-ref.ids" "$O/gen-mut-$m.ids"; then v="ids SAME as the clean tp=2 run"; else v="ids DIFFER"; fi
    log "MUTANT $m: $v; $(grep -ahE 'tensor-parallel rank 1: decode graph|^Error|trap|Aborted|core dumped|HSA_STATUS' "$O/gen-mut-$m.err" | head -3 | tr '\n' '|')"
  done
fi
log "done, time left $(left)s"
