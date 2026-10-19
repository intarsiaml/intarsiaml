#!/usr/bin/env bash
# Core-op fusion phase 2 on tatooine (docs/results/2026-10-19-core-op-fusion-p2.md). Runs ON tatooine inside a fleetq lease.
#   CPU-only slot (--mem 0, radiance stays up):   fusion-p2.sh mutbuild        (the p2-* and the 4 round-6 moe_wmma.h mutant test binaries)
#   GPU window (--takes-down radiance, burst.sh):  fusion-p2.sh tests mutrun    (bitwise tests, every mutant)
# Unlike remote.sh this script never touches a container it did not start, so it is safe in a CPU slot that runs beside
# another job's GPU window. Trees: INTARSIA_REPO (default ~/intarsiaml-fp2, cargo volume intarsia-cargo-target-fp2); base f3cbba21 in ~/intarsiaml-fp2base.
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-fp2}"
export INTARSIA_TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-fp2}"
export BASE_REPO="${BASE_REPO:-$HOME/intarsiaml-fp2base}" BASE_VOL="${BASE_VOL:-intarsia-cargo-target-fp2base}"
cd "$INTARSIA_REPO"
D=deploy/tatooine/dev.sh
P2MUT="p2-ga-gate p2-ga-qstride p2-ga-woff p2-ga-fp8 p2-ga-vpos p2-cd-writer0 p2-cd-noreset p2-cd-novstate p2-cd-round p2-cd-tap p2-cd-contract p2-cd-hist p2-cd-slotstate"  # p2-sel-m33 is a lib test (no GPU): run on the Mac
# the round-6 moe_wmma.h mutants re-enabled by the mutate.py occurrence index (they matched twice since round 6)
R6MUT="moe-z-step pairs-sred pairs-xrow pairs-out"
ts() { date +%T; }

stage_mutbuild() {  # CPU only: each mutant's test binary, self-contained (static kernel library), into /target/p2mut/<name>
  local MUT="${INTARSIA_MUT_DIR:-$HOME/intarsiaml-fp2-mut}" m t exe
  for m in ${MUTATIONS:-$P2MUT $R6MUT}; do
    rm -rf "$MUT"; mkdir -p "$MUT"
    rsync -a --exclude target --exclude bench-out --exclude prof-out "$INTARSIA_REPO/" "$MUT/"
    python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" "$MUT" "$m" || continue
    t=$(python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" x --test-of "$m")
    exe=$(INTARSIA_REPO="$MUT" $D cargo test --release -p intarsia-backend-hip --test "$t" --no-run --message-format=json 2>/dev/null \
      | python3 -c 'import json,sys
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    if d.get("reason") == "compiler-artifact" and d.get("executable") and d["target"]["kind"] == ["test"]: print(d["executable"])' | tail -1)
    # a lib-test mutant's binary is the crate's unit-test target ("lib"), not a tests/ file
    if [ "$t" = lib ]; then
      exe=$(INTARSIA_REPO="$MUT" $D cargo test --release -p intarsia-backend-hip --lib --no-run --message-format=json 2>/dev/null \
        | python3 -c 'import json,sys
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    if d.get("reason") == "compiler-artifact" and d.get("executable") and d["profile"]["test"]: print(d["executable"])' | tail -1)
    fi
    if [ -z "$exe" ]; then echo "$m: BUILD FAILED"; continue; fi
    INTARSIA_REPO="$MUT" $D bash -c "mkdir -p /target/p2mut && cp '$exe' /target/p2mut/$m"
    echo "$m: test $t built [$(ts)]"
  done
  rm -rf "$MUT"
  # the mutant builds shared this tree's cargo volume: make the real tree recompile its kernels on the next build
  find "$INTARSIA_REPO/kernels" "$INTARSIA_REPO/crates" -type f -exec touch {} +
  echo "NOTE: rebuild this tree (scripts/hip-decode/build.sh) before running binaries from its target volume"
}

stage_mutrun() {  # GPU: every mutant binary must FAIL (or hang -> killed by the timeout)
  local m
  for m in ${MUTATIONS:-$P2MUT $R6MUT}; do
    printf '%-16s ' "$m"
    local to=${MUT_TIMEOUT:-120}
    HIP_VISIBLE_DEVICES=${MUT_GPU:-0} $D bash -c "timeout -s KILL $to /target/p2mut/$m --test-threads=1; echo exit=\$?" 2>&1 \
      | grep -aE "^test result|panicked at|^exit=|launches, " | head -4 | cut -c1-200 | tr '\n' ' '
    echo
  done
}

stage_tests() {  # GPU: the phase-2 bitwise tests (every prefetch depth), the suites of the kernels the folds mirror, the lib pins
  local pd
  for pd in 1 2 4; do
    echo "--- fusion_p2 INTARSIA_HIP_GDN_PD=$pd"
    HIP_VISIBLE_DEVICES=${TEST_GPU:-0} INTARSIA_HIP_GDN_PD=$pd timeout 300 $D cargo test --release -p intarsia-backend-hip --test fusion_p2 \
      -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|FAILED|panicked|differ|cases ran" | cut -c1-300 | tail -12
  done
  HIP_VISIBLE_DEVICES=${TEST_GPU:-0} timeout 600 $D cargo test --release -p intarsia-backend-hip --no-fail-fast --lib --test fusion_p1 \
    --test gdn_v2 --test qk_rope_kv --test moe_pairs --test quantmoe --test fused_decode --test decode_v2 --test device_pos -- --nocapture --test-threads=1 2>&1 \
    | grep -aE "^test |test result|FAILED|panicked|differ|cases bitwise|launches, " | cut -c1-300 | tail -80
}

stage_alltests() {  # GPU: the whole HIP crate
  HIP_VISIBLE_DEVICES=${TEST_GPU:-0} timeout 900 $D cargo test --release -p intarsia-backend-hip --no-fail-fast -- --test-threads=1 2>&1 \
    | grep -aE "test result|FAILED|panicked|Running" | cut -c1-200 | tail -60
}

M35=/models/Qwen3.6-35B-A3B-NVFP4
M8=/models/Qwen3-8B
B=/target/release/intarsia
stage_plans() {  # GPU (TEST_GPU): plan show --json of both trees for the 35B and the 8B, kernel-per-op diff, calls per decode step
  local O="$INTARSIA_REPO/bench-out/plans" t m
  mkdir -p "$O"
  for t in fp2 base; do
    for m in 35b 8b; do
      local repo="$INTARSIA_REPO" vol="$INTARSIA_TARGET_VOL" mp=$M35 extra="--mem-budget 29"
      [ $t = base ] && repo="$BASE_REPO" && vol="$BASE_VOL"
      [ $m = 8b ] && mp=$M8 && extra=""
      INTARSIA_REPO="$repo" INTARSIA_TARGET_VOL="$vol" HIP_VISIBLE_DEVICES=${TEST_GPU:-0} \
        $D $B plan show --backend hip --model $mp $extra --max-seq 4096 --json > "$O/$t-$m.json" 2> "$O/$t-$m.err"
      echo "plan $t $m: exit $? $(wc -c < "$O/$t-$m.json") bytes"
    done
  done
  python3 "$INTARSIA_REPO/scripts/hip-decode/plan_diff.py" "$O" fp2 base 2>&1 | tee "$O/diff.txt"
}

stage_adm() {  # GPU (TEST_GPU): FULL admission sweeps (no --quick) of the 35B and the 8B shapes, branch and base tree
  local O="$INTARSIA_REPO/bench-out/adm" t m
  mkdir -p "$O"
  for t in fp2 base; do
    for m in 35b 8b; do
      local repo="$INTARSIA_REPO" vol="$INTARSIA_TARGET_VOL" mp=$M35
      [ $t = base ] && repo="$BASE_REPO" && vol="$BASE_VOL"
      [ $m = 8b ] && mp=$M8
      INTARSIA_REPO="$repo" INTARSIA_TARGET_VOL="$vol" HIP_VISIBLE_DEVICES=${TEST_GPU:-0} \
        $D $B kernels test --backend hip --model $mp > "$O/$t-$m.log" 2>&1
      echo "adm $t $m exit $?: $(grep -a ' kernels, ' "$O/$t-$m.log" | tail -1) | FAIL/REJECT lines: $(grep -acE 'FAIL|REJECT' "$O/$t-$m.log")"
    done
  done
  grep -aE "gate_split_qk_norm_rope_kv|gdn_conv_delta" "$O/fp2-35b.log" | cut -c1-200 | head -12
}

# ids4k with the two phase-2 folds switched off through INTARSIA_HIP_DISABLE (the kill switch), against the base tree's ids
stage_idsoff() {
  IDS_NEW_ENV="INTARSIA_HIP_DISABLE=gate_split_qk_norm_rope_kv_bf16,gdn_conv_delta_bf16" \
    bash "$INTARSIA_REPO/scripts/hip-decode/remote.sh" ids4k
}

for s in "$@"; do
  echo "################ stage $s start $(ts)"
  if [[ "$s" == r:* ]]; then
    bash "$INTARSIA_REPO/scripts/hip-decode/remote.sh" "${s#r:}"
  else
    "stage_$s"
  fi
  echo "################ stage $s end $(ts)"
done
