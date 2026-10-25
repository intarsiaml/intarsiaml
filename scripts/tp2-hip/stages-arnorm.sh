#!/usr/bin/env bash
# Lane tp2 (ar-norm) window (ON tatooine, radiance down, called by window.sh: W=arnorm): the fused all-reduce + residual add +
# rmsnorm (kernels/hip/ar_norm.hip, INTARSIA_TP_ARNORM) against the unfused sequence. Budget CAP seconds (default 2100 = 35
# min); every stage first checks the time left and is skipped (logged "SKIPPED") when it no longer fits, so the order is the
# priority. Needs, built in the CPU slot beforehand (scripts/tp2-hip/sync.sh, build.sh with HIPTEST=1, arnorm-mutants.sh):
#   trees  arnorm (this branch, CLI /work/tp2-intarsia, HIP tests in volume intarsia-cargo-target-tp2-arnorm-hiptest),
#          base (origin/main, CLI built), arnorm-mut (the mutant test binaries in mutbin/)
# Where the fusion fires: only where the norm after a block-end all-reduce is a standalone RmsNormAdd. The 35B folds the norm
# into the next block's GEMV pair on 79 of its 80 blocks (op counts in docs/results/2026-10-21-h1-tp2-hip), so it fuses ONE
# all-reduce per decode step (the final norm); the dense 8B fuses all 72. The 8B is the model this stage can show a speed
# effect on; the 35B is the identity check next to the folds. State the measured fused counts, not the lane brief's.
#
# Checks (log lines are SAME / DIFF / KILLED / SURVIVED / SKIPPED, the numbers come from the plan and the rank counters):
#   1. HIP GPU tests: tests/ar_norm.rs (fused == the unfused pair bit for bit on wide-range data, all hidden sizes, rows,
#      in place, mixed with plain collectives eagerly and in a replayed graph) and the comm tests.
#   2. mutants of ar_norm.hip (unrounded all-reduce, unrounded sum, other block size, w_offset ignored, residual dropped):
#      each must FAIL the ar_norm test (KILLED).
#   3. plan show at tp=2: the fused count per decode step (8B 72, 35B 1) with INTARSIA_TP_ARNORM=1, 0 with =0; tp=1 plan
#      identical to origin/main.
#   4. 256 greedy ids, tp=2, ARNORM=1 against ARNORM=0: 8B and 35B, short and 4k; the ranks' "fused all-reduce+rmsnorm" counters
#      (eager and captured in the decode graph) are > 0 on the =1 runs (a run that fused nothing is VACUOUS, not a pass).
#   5. tp=1 35B 256 ids, arnorm tree (ARNORM=1 asked for) against origin/main.
#   6. accuracy dumps tp=2, =1 against =0: 8B short and 35B short, bitwise (cmp; scripts/tp2-hip/cmpdump.py is the offline
#      authority) and the self-golden gate with the floor budgets.
#   7. decode speed, `bench speed` 1k / 4k, gen 128, tp=2 graphs + fusion: ARNORM=0 / 1 interleaved, 8B (2 rounds) and 35B (1).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
CAP=${CAP:-2100}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { # need <seconds> <what>: succeed when the window still has that long, else log SKIPPED
  if (( $(left) > $1 )); then return 0; fi
  log "SKIPPED (time left $(left)s < $1): $2"; return 1
}
A0="INTARSIA_TP_ARNORM=0"; A1="INTARSIA_TP_ARNORM=1"
G8=$G/qwen3-8b.safetensors
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
VOLT=intarsia-cargo-target-tp2-arnorm-hiptest
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8"
log "window start, CAP ${CAP}s"

# fused <tag>: the fused count per decode step in a plan JSON
fused() { grep -aoE '"all_reduce_fused_rmsnorm_per_decode_step": *[0-9]+' "$O/plan-$1.out" | grep -oE '[0-9]+$' | head -1; }
# evidence <tag>: the ranks' fused-collective counters of a run, and a VACUOUS line when an =1 run fused nothing
evidence() {
  local l; l=$(grep -ah 'fused all-reduce+rmsnorm' "$O/gen-$1.err" "$O/self-$1.err" "$O/acc-$1.err" 2>/dev/null | tr '\n' '|')
  log "evidence $1: ${l:-none printed}"
  [[ $2 == on ]] && ! grep -qaE 'eager=[1-9]|captured=[1-9]' <<<"$l" && log "VACUOUS $1: no fused collective ran"
  return 0
}

# ---- 1. HIP GPU tests (the arnorm hiptest volume was built in the CPU slot)
ht() { # ht <tag> <cargo test args...>
  local tag=$1; shift
  INTARSIA_REPO=$(T arnorm) INTARSIA_TARGET_VOL=$VOLT timeout 420 "$(T arnorm)/deploy/tatooine/dev.sh" \
    cargo test --release -p intarsia-backend-hip --no-fail-fast "$@" > "$O/hiptest-$tag.log" 2>&1 < /dev/null
  log "hiptest $tag exit $?: $(grep -aE '^test result' "$O/hiptest-$tag.log" | awk '{p+=$4; f+=$6; i+=$8} END {print p " passed, " f " failed, " i " ignored"}') $(grep -aE 'FAILED|panicked|HSA_STATUS|^[0-9]+ trials' "$O/hiptest-$tag.log" | head -4 | tr '\n' '|')"
}
need 240 "hiptest ar_norm" && ht arnorm --test ar_norm -- --nocapture
need 200 "hiptest comm" && ht comm --test comm

# ---- 3. plans (cheap) before the long runs
if need 400 "plans"; then
  E=$A1 plan arnorm 0,1 tp2-8b-on $M8 $TP2;  log "fused all-reduces per decode step, 8B tp=2 ARNORM=1: $(fused tp2-8b-on) (want 72)"
  E=$A0 plan arnorm 0,1 tp2-8b-off $M8 $TP2; log "fused all-reduces per decode step, 8B tp=2 ARNORM=0: $(fused tp2-8b-off) (want 0)"
  E=$A1 plan arnorm 0,1 tp2-35b-on $M35 $TP2; log "fused all-reduces per decode step, 35B tp=2 ARNORM=1: $(fused tp2-35b-on) (want 1: the final norm; the rest are folded)"
  E=$A1 plan arnorm 0 tp1-35b-new $M35;     plan base 0 tp1-35b-base $M35
  same plan-tp1-35b-new.out plan-tp1-35b-base.out "tp=1 35B plan, arnorm tree (ARNORM=1 asked) vs origin/main"
fi

# ---- 4. ids, short first
ids() { # ids <model var name> <label> <kind>  -> tag-on / tag-off, SAME/DIFF
  local m=$2 lab=$1 kind=$3 s=${3:0:1}
  E=$A1 gen arnorm 0,1 "$lab-$s-on" "$m" "$kind" $TP2;  evidence "$lab-$s-on" on
  E=$A0 gen arnorm 0,1 "$lab-$s-off" "$m" "$kind" $TP2; evidence "$lab-$s-off" off
  same "gen-$lab-$s-on.ids" "gen-$lab-$s-off.ids" "$lab tp=2 $kind 256 ids, ARNORM=1 vs 0"
}
need 200 "8B short ids" && ids 8b $M8 short
need 330 "35B short ids" && ids 35b $M35 short

# ---- 2. mutants (the binaries were built in the CPU slot: arnorm-mutants.sh)
if need 420 "mutants" && ls "$(T arnorm-mut)"/mutbin/* >/dev/null 2>&1; then
  for b in "$(T arnorm-mut)"/mutbin/*; do
    n=$(basename "$b")
    INTARSIA_REPO=$(T arnorm-mut) timeout 200 "$(T arnorm-mut)/deploy/tatooine/dev.sh" "/work/mutbin/$n" --test-threads 1 \
      fused_is_bitwise bf16_partials > "$O/mutant-$n.log" 2>&1 < /dev/null
    rc=$?
    if (( rc != 0 )); then log "KILLED mutant $n (exit $rc): $(grep -aE 'panicked|assertion|vs unfused|vs host' "$O/mutant-$n.log" | head -2 | cut -c1-160 | tr '\n' '|')"
    else log "SURVIVED mutant $n (the ar_norm test passed on the mutated kernel)"; fi
  done
fi

# ---- 6. accuracy dumps, =1 against =0 (8B first: the model the fusion fires on)
self8() { # self8 <tag> <dump in $O> [args]: change vs a dump of the 8B
  local tag=$1 d=$2; shift 2
  cp "$O/$d" "$(T arnorm)/tp2-out/$d"
  run arnorm 0,1 420 "self-$tag" accuracy --backend hip --model $M8 --mem-budget 29 --max-seq 17408 \
    --self-golden /work/tp2-out/$d --mode both --allow-same-code --allow-config-mismatch $TP2 "$@"
}
if need 480 "8B accuracy dumps"; then
  E=$A0 acc arnorm 0,1 8b-off $M8 $G8 $TP2
  E=$A1 self8 8b-on acc-8b-off.safetensors $FLOOR --write-self-golden /work/tp2-out/acc-8b-on.safetensors
  mv "$(T arnorm)/tp2-out/acc-8b-on.safetensors" "$O/" 2>/dev/null; evidence 8b-on on
  same acc-8b-off.safetensors acc-8b-on.safetensors "8B tp=2 accuracy dump ARNORM=1 vs 0 (byte cmp)"
fi
if need 600 "35B accuracy dumps"; then
  E=$A0 acc arnorm 0,1 35b-off $M35 $SG $TP2
  E=$A1 self arnorm 0,1 35b-on acc-35b-off.safetensors $TP2 $FLOOR --write-self-golden /work/tp2-out/acc-35b-on.safetensors
  mv "$(T arnorm)/tp2-out/acc-35b-on.safetensors" "$O/" 2>/dev/null; evidence 35b-on on
  same acc-35b-off.safetensors acc-35b-on.safetensors "35B tp=2 accuracy dump ARNORM=1 vs 0 (byte cmp)"
fi

# ---- 7. speed: one fresh server per arm, tp=2 graphs + fusion, ARNORM 0 / 1
srv() { # srv <arm 0|1> <model path> <name> <max-seq>
  docker rm -f arnorm-serve >/dev/null 2>&1
  docker run -d --name arnorm-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
    --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
    -e HIP_VISIBLE_DEVICES=0,1 -e INTARSIA_TP_ARNORM="$1" -v "$(T arnorm)":/work -v /data/models/intarsia:/models:ro -w /work "$IMG" \
    /work/tp2-intarsia serve --backend hip --model "$2" --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port 8092 \
    --served-model-name "$3" --max-seq "$4" $TP2 > /dev/null < /dev/null
  for i in $(seq 1 80); do curl -sf -m 2 http://localhost:8092/health > /dev/null 2>&1 < /dev/null && return 0; sleep 3; done
  docker logs --tail 30 arnorm-serve > "$O/serve-fail.log" 2>&1; return 1
}
speed() { # speed <label> <arm> <round> <model path> <name> <max-seq> <sizes>
  local l=ar-$1-a$2-r$3
  srv "$2" "$4" "$5" "$6" || { log "$l: server did not come up"; return 1; }
  docker exec arnorm-serve /work/tp2-intarsia bench speed --url http://localhost:8092 --model "$5" --sizes "$7" --gen 128 --reps 1 \
    --warmup 1 --at-most --thinking off --require-graphs --label "$l" --out-dir /work/tp2-out/bench > "$O/bench-$l.log" 2>&1 < /dev/null
  log "$l exit $?: $(grep -aE '^ *(1024|4096) |require-graphs' "$O/bench-$l.log" | tr -s ' ' | tr '\n' '|')"
  docker logs arnorm-serve > "$O/serve-$l.log" 2>&1
  docker rm -f arnorm-serve > /dev/null 2>&1
  cp "$(T arnorm)/tp2-out/bench/$l.json" "$O/" 2>/dev/null
}
mkdir -p "$(T arnorm)/tp2-out/bench"
if need 420 "8B speed A/B"; then
  for r in 1 2; do
    arms="0 1"; [[ $r == 2 ]] && arms="1 0"
    for a in $arms; do speed 8b $a $r /models/Qwen3-8B qwen3-8b-ar 8192 1024,4096; done
  done
fi
if need 330 "35B speed A/B"; then
  for a in 0 1; do speed 35b $a 1 $M35 qwen36-35b-ar 8192 1024,4096; done
fi

# ---- the rest of the ids, tp=1, 4k
if need 330 "35B tp=1 ids, arnorm tree vs origin/main"; then
  E=$A1 gen arnorm 0 tp1-new $M35 short; gen base 0 tp1-base $M35 short
  same gen-tp1-new.ids gen-tp1-base.ids "tp=1 35B 256 ids, arnorm tree (ARNORM=1 asked) vs origin/main"
fi
need 200 "8B 4k ids" && ids 8b $M8 4k
need 330 "35B 4k ids" && ids 35b $M35 4k
if need 240 "kernels test"; then
  run arnorm 0 240 kernels-test kernels test --backend hip --device 0 --model $M35
  log "kernels test: $(grep -aiE 'FAIL|kernels,|cases' "$O/kernels-test.out" | tail -3 | tr '\n' '|')"
fi
log "done, $(left)s left"
