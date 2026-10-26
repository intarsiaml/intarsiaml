#!/usr/bin/env bash
# Lane tp2 final gates on the merged head (ON tatooine, radiance down, called by window.sh). Trees: D = the lane head merged with origin/main
# (tree `lock`, scripts/tp2-hip/sync.sh lock HEAD; build.sh HIPTEST=1 HIPTEST_MODELS=1), B = origin/main (tree `base`).
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 42 --mem 60 --label "tp2 final F1" -- \
#     env W=f1 CAP=2350 XENV="D=lock B=base PC=<head sha> STAGE=1" scripts/tp2-hip/window.sh
# STAGE=1 (35B): tp=1 no-op against main (plan show + 256 ids, 8B and 35B, short and 4k; D on card 0 and B on card 1 at the same time),
#   the whole non-ignored batch_cuda_real file at tp=2 (batched == solo, step-protocol mutants, 8B, engine == local engine),
#   then card 0: batch_recurrent --test-threads=1 and card 1: kernels test --quick.
# STAGE=2: 35B tp=2 gates (run to run at floor budgets, distance to vLLM against tp=1, prefix reuse, 256 ids short/4k run to run),
#   27B batched == solo at tp=2, the whole non-ignored batch_cuda_real file at tp=1 (card 0).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-lock}; B=${B:-base}; PC=${PC:?PC=<head sha of the build>}; STAGE=${STAGE:-1}
CAP=${CAP:-2350}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { if (( $(left) > $1 )); then return 0; fi; log "SKIPPED (time left $(left)s < $1): $2"; return 1; }
vram() { local m=0 v c; for c in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do v=$(cat "$c" 2>/dev/null || echo 0); (( v > m )) && m=$v; done; echo $m; }
clear() { local i; for i in 1 2 3 4 5 6; do (( $(vram) < 1073741824 )) && return 0; sleep 10; done
  log "CONTENDED before '$1': $(( $(vram) / 1048576 )) MiB in use on a card: NO VERDICT"; return 1; }
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8 --pre-change $PC"
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
HUB=/data/models/radiance/hf-cache/hub
log "final stage $STAGE start, CAP ${CAP}s, pre-change $PC, VRAM in use $(( $(vram) / 1048576 )) MiB"
mkdir -p "$(T $D)/tp2-out" "$(T $B)/tp2-out"
for t in $D $B; do echo "$t: $(cat "$(T $t)/tp2-intarsia.stamp" 2>/dev/null || ls -la "$(T $t)/tp2-intarsia")"; done > "$O/versions.txt"

# mtest <tag> <gpus> <timeout> <cargo test args...>: intarsia-models GPU tests, prebuilt in the -hiptest volume
mtest() {
  local tag=$1 gpus=$2 to=$3; shift 3
  # shellcheck disable=SC2086
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest HIP_VISIBLE_DEVICES=$gpus timeout "$to" \
    "$(T $D)/deploy/tatooine/dev.sh" env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=${MQ:-$M35} INTARSIA_QWEN3_DIR=$M8 ${E:-} \
    cargo test --release -p intarsia-models "$@" > "$O/mtest-$tag.log" 2>&1 < /dev/null
  local rc=$?
  log "mtest $tag exit $rc${E:+ [$E]}: $(grep -aE '^test result|FAILED|panicked|SKIPPED|pairs bitwise|mutant .* caught|tensor-parallel rank 1: batched' "$O/mtest-$tag.log" | head -14 | cut -c1-170 | tr '\n' '|')"
  return $rc
}

if [[ $STAGE == 1 ]]; then
  # 1. tp=1 no-op against main: D on card 0, B on card 1, same commands
  if clear "noop"; then
    noop() { # noop <tree> <gpu> <tag>
      local t=$1 g=$2 tag=$3 m mm kind
      for m in 35 8; do
        mm=$M35; [[ $m == 8 ]] && mm=$M8
        for kind in short 4k; do gen $t $g "$tag-$m-$kind" $mm $kind; done
        plan $t $g "$tag-$m" $mm
      done
    }
    noop $D 0 d & noop $B 1 b & wait
    for m in 35 8; do
      for kind in short 4k; do same "gen-d-$m-$kind.ids" "gen-b-$m-$kind.ids" "tp=1 $m $kind 256 ids, head vs main"; done
      # the plan json carries the build stamp: compare without it
      if diff <(grep -av 'stamp\|head=\|code-tree' "$O/plan-d-$m.out") <(grep -av 'stamp\|head=\|code-tree' "$O/plan-b-$m.out") > "$O/plan-$m.diff"; then log "SAME plan show $m (json)"; else log "DIFF plan show $m: $(wc -l < "$O/plan-$m.diff") lines in plan-$m.diff"; fi
    done
  fi
  # 2. the whole non-ignored batch_cuda_real file at tp=2 (N = 2..16 x 130 steps, mutants, 8B, engine == local engine)
  need 1300 "batch_cuda_real tp2" && clear "batch_cuda_real tp2" && E="INTARSIA_BATCH_TEST_TP=2" mtest tp2-all 0,1 1300 --test batch_cuda_real -- --test-threads=1 --nocapture
  # 3. card 0: tp=1 batch_recurrent; card 1: kernels test --quick (the binary, not cargo)
  if need 500 "batch_recurrent + kernels"; then
    clear "batch_recurrent" && mtest tp1-batch-recurrent 0 600 --test batch_recurrent -- --test-threads=1 &
    run $D 1 420 kernels-test-quick kernels test --backend hip --device 0 --model $M35 --quick
    wait
    log "kernels test --quick: $(grep -aiE 'cases|fail' "$O/kernels-test-quick.out" | tail -2 | tr '\n' '|')"
  fi
elif [[ $STAGE == 3 ]]; then
  # F3: the direct partial-sum check and the whole HIP crate suite, the comm/sharding mutants, KL of tp=2 against tp=1 (report-only in the agreed gate)
  ht() { # ht <tag> <cargo test args...>
    local tag=$1; shift
    INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest timeout 600 "$(T $D)/deploy/tatooine/dev.sh" \
      cargo test --release -p intarsia-backend-hip --no-fail-fast "$@" > "$O/hiptest-$tag.log" 2>&1 < /dev/null
    log "hiptest $tag exit $?: $(grep -aE '^test result' "$O/hiptest-$tag.log" | awk '{p+=$4; f+=$6; i+=$8} END {print p " passed, " f " failed, " i " ignored"}') $(grep -aE 'FAILED|panicked|HSA_STATUS' "$O/hiptest-$tag.log" | head -4 | tr '\n' '|')"
  }
  clear "tp_partial_sum" && ht tp-partial-sum --test tp_partial_sum -- --nocapture --test-threads=1
  need 600 "all hip tests" && clear "all hip tests" && ht all
  if need 900 "mutants" && clear "mutants"; then
    gen $D 0,1 mut-ref $M35 short $TP2
    for m in no-peer-upload drop-captured-allreduce peer-ignores-flag tp-fold-outproj batch-peer-no-table vocab-no-gather; do
      need 90 "mutant $m" || continue
      [[ -x $(T mut-$m)/tp2-intarsia ]] || { log "MUTANT $m: no binary"; continue; }
      gen mut-$m 0,1 mut-$m $M35 short $TP2
      if cmp -s "$O/gen-mut-ref.ids" "$O/gen-mut-$m.ids"; then v="ids SAME as the clean tp=2 run (SURVIVED)"; else v="ids DIFFER (killed)"; fi
      log "MUTANT $m: $v; $(grep -ahE 'tensor-parallel rank 1: decode graph|^Error|trap|Aborted|core dumped|HSA_STATUS' "$O/gen-mut-$m.err" | head -3 | tr '\n' '|')"
    done
    MT=mut-batch-peer-no-table
    INTARSIA_REPO=$(T $MT) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-mutb-hiptest HIP_VISIBLE_DEVICES=0,1 timeout 540 \
      "$(T $MT)/deploy/tatooine/dev.sh" env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 INTARSIA_BATCH_TEST_TP=2 \
      cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture --exact real_35b_batched_decode_is_bitwise_the_solo_step \
      > "$O/mtest-mut-batch-peer-no-table.log" 2>&1 < /dev/null
    rc=$?
    if grep -aq 'SKIPPED' "$O/mtest-mut-batch-peer-no-table.log"; then v="NO VERDICT (the test skipped)"; elif (( rc == 124 )); then v="TIMEOUT"; elif (( rc != 0 )); then v="KILLED (exit $rc)"; else v="SURVIVED"; fi
    log "MUTANT batch-peer-no-table (batched test): $v: $(grep -aE 'panicked|differ|HSA_STATUS' "$O/mtest-mut-batch-peer-no-table.log" | head -2 | cut -c1-200 | tr '\n' '|')"
  fi
  if need 700 "KL tp2 vs tp1" && clear "KL tp2 vs tp1"; then
    acc $D 0 ext-tp1 $M35 $SG
    run $D 0,1 420 self-tp2-vs-tp1-short accuracy --backend hip --model $M35 --mem-budget 29 --max-seq 17408 --self-golden /work/tp2-out/acc-ext-tp1.safetensors \
      --mode both --allow-same-code --allow-config-mismatch $TP2 --self-vs-ref-noise --ref-noise $NS --ref-noise-factor 1.0 --pre-change $PC
    run $D 0 600 acc-tp1-35b-long accuracy --backend hip --model $M35 --golden $LG --mode both --mem-budget 29 --max-seq 17408 --write-self-golden /work/tp2-out/acc-tp1-35b-long.safetensors
    run $D 0,1 600 self-tp2-vs-tp1-long accuracy --backend hip --model $M35 --mem-budget 29 --max-seq 17408 --self-golden /work/tp2-out/acc-tp1-35b-long.safetensors \
      --mode both --allow-same-code --allow-config-mismatch $TP2 --self-vs-ref-noise --ref-noise $NL --ref-noise-factor 1.0 --pre-change $PC
  fi
else
  # 1. 35B tp=2: run to run at floor budgets, tp=1 and tp=2 against vLLM (check 3), prefix reuse, ids short and 4k run to run
  if clear "35B tp2"; then
    acc $D 0 ext-tp1 $M35 $SG
    acc $D 0,1 ext-tp2 $M35 $SG $TP2
    acc $D 0,1 ext-tp2-b $M35 $SG $TP2
    self $D 0,1 tp2-b-vs-a acc-ext-tp2.safetensors $TP2 $FLOOR
    python3 "$(T $D)/scripts/tp2-hip/tp-gate.py" --ref "$O/acc-ext-tp1.out" --cand "$O/acc-ext-tp2.out" > "$O/gate-check3.txt" 2>&1
    log "check 3 (tp-gate.py) exit $?: $(grep -aE '^(prefill|decode)|check 3' "$O/gate-check3.txt" | tr '\n' '|')"
  fi
  if need 500 "reuse + ids"; then
    run $D 0,1 400 reuse-tp2 accuracy --backend hip --model $M35 --golden $LG --mode reuse --mem-budget 29 --max-seq 17408 $TP2
    gen $D 0,1 tp2-a-short $M35 short $TP2; gen $D 0,1 tp2-b-short $M35 short $TP2
    gen $D 0,1 tp2-a-4k $M35 4k $TP2; gen $D 0,1 tp2-b-4k $M35 4k $TP2
    gen $D 0 tp1-short $M35 short; gen $D 0 tp1-4k $M35 4k
    same gen-tp2-a-short.ids gen-tp2-b-short.ids "35B tp=2 short run to run"; same gen-tp2-a-4k.ids gen-tp2-b-4k.ids "35B tp=2 4k run to run"
    same gen-tp2-a-short.ids gen-tp1-short.ids "35B tp=2 vs tp=1 short"; same gen-tp2-a-4k.ids gen-tp1-4k.ids "35B tp=2 vs tp=1 4k"
  fi
  # 2. 27B batched == solo at tp=2
  if need 900 "27B batched == solo at tp2" && clear "27B"; then
    INTARSIA_MODELS=$HUB MQ=$M27 E="INTARSIA_BATCH_TEST_TP=2" mtest tp2-27b-batched 0,1 840 --test batch_cuda_real -- --test-threads=1 --nocapture --exact real_35b_batched_decode_is_bitwise_the_solo_step
  fi
  # 3. the whole non-ignored batch_cuda_real file at tp=1 (card 0)
  need 900 "batch_cuda_real tp1" && clear "batch_cuda_real tp1" && mtest tp1-all 0 900 --test batch_cuda_real -- --test-threads=1 --nocapture
fi
log "final stage $STAGE done, time left $(left)s"
