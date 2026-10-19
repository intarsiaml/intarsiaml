#!/usr/bin/env bash
# Core-op fusion phase 1 on tatooine (docs/results/2026-10-19-core-op-fusion-p1.md). Runs ON tatooine inside a fleetq lease.
#   CPU-only slot (--mem 0, radiance stays up):   fusion-p1.sh mutbuild        (the p1-* and v3-* mutant test binaries)
#   GPU window (--takes-down radiance, burst.sh):  fusion-p1.sh tests mutrun    (bitwise tests, stress, every mutant)
# Unlike remote.sh this script never touches a container it did not start, so it is safe in a CPU slot that runs beside
# another job's GPU window. Trees: INTARSIA_REPO (default ~/intarsiaml-fusion, cargo volume intarsia-cargo-target-fusion).
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-fusion}"
export INTARSIA_TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-fusion}"
export BASE_REPO="${BASE_REPO:-$HOME/intarsiaml-fusionbase}" BASE_VOL="${BASE_VOL:-intarsia-cargo-target-fusionbase}"
cd "$INTARSIA_REPO"
D=deploy/tatooine/dev.sh
P1MUT="p1-n5-sc0 p1-n5-nowrite p1-n5-round p1-n5-woff p1-n5-stage p1-g4-sig p1-g4-gate p1-dc-logit0 p1-dc-sground p1-dc-soround p1-dc-tail p1-dc-row"
# v3-ticket hangs the kernel by design: last, so a wedged queue cannot disturb the others
V3MUT="v3-noreset v3-early v3-nowait v3-hrow v3-gurow v3-nofence v3-ticket"
ts() { date +%T; }

stage_mutbuild() {  # CPU only: each mutant's test binary, self-contained (static kernel library), into /target/p1mut/<name>
  local MUT="${INTARSIA_MUT_DIR:-$HOME/intarsiaml-fusion-mut}" m t exe
  for m in ${MUTATIONS:-$P1MUT $V3MUT}; do
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
    if [ -z "$exe" ]; then echo "$m: BUILD FAILED"; continue; fi
    INTARSIA_REPO="$MUT" $D bash -c "mkdir -p /target/p1mut && cp '$exe' /target/p1mut/$m"
    echo "$m: test $t built [$(ts)]"
  done
  rm -rf "$MUT"
  # the mutant builds shared this tree's cargo volume: make the real tree recompile its kernels on the next build
  find "$INTARSIA_REPO/kernels" "$INTARSIA_REPO/crates" -type f -exec touch {} +
  echo "NOTE: rebuild this tree (scripts/hip-decode/build.sh) before running binaries from its target volume"
}

stage_mutrun() {  # GPU: every mutant binary must FAIL (or hang -> killed by the timeout)
  local m
  for m in ${MUTATIONS:-$P1MUT $V3MUT}; do
    printf '%-16s ' "$m"
    HIP_VISIBLE_DEVICES=${MUT_GPU:-0} $D bash -c "timeout -s KILL ${MUT_TIMEOUT:-150} /target/p1mut/$m --test-threads=1; echo exit=\$?" 2>&1 \
      | grep -aE "^test result|panicked at|^exit=|launches, " | head -4 | cut -c1-200 | tr '\n' ' '
    echo
  done
}

stage_tests() {  # GPU: the phase-1 bitwise tests, the v3 tests (sentinel stress included) and the fused / decode suites
  HIP_VISIBLE_DEVICES=${TEST_GPU:-0} timeout 600 $D cargo test --release -p intarsia-backend-hip --no-fail-fast --test fusion_p1 --test moe_v3 \
    --test fused_decode --test decode_v2 --test device_pos -- --nocapture --test-threads=1 2>&1 \
    | grep -aE "^test |test result|FAILED|panicked|differ|cases bitwise|launches, " | cut -c1-300 | tail -60
}

stage_alltests() {  # GPU: the whole HIP crate
  HIP_VISIBLE_DEVICES=${TEST_GPU:-0} timeout 900 $D cargo test --release -p intarsia-backend-hip --no-fail-fast -- --test-threads=1 2>&1 \
    | grep -aE "test result|FAILED|panicked|Running" | cut -c1-200 | tail -60
}

stage_stress() {  # GPU: the sentinel stress alone, more rounds (INTARSIA_HIP_V3_STRESS_ROUNDS), on the real tree
  HIP_VISIBLE_DEVICES=${TEST_GPU:-0} INTARSIA_HIP_V3_STRESS_ROUNDS=${ROUNDS:-600} timeout 600 $D cargo test --release -p intarsia-backend-hip --test moe_v3 \
    v3_sentinel_stress -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|panicked|launches, |differs" | cut -c1-300 | tail -20
}

stage_nofencestress() {  # GPU: the v3-nofence mutant binary on the sentinel stress only, more rounds
  HIP_VISIBLE_DEVICES=${MUT_GPU:-0} $D bash -c "INTARSIA_HIP_V3_STRESS_ROUNDS=${ROUNDS:-600} timeout -s KILL 500 /target/p1mut/v3-nofence v3_sentinel_stress --nocapture --test-threads=1; echo exit=\$?" 2>&1 \
    | grep -aE "^test |test result|panicked|launches, |differs|^exit=" | cut -c1-300 | tail -20
}

M35=/models/Qwen3.6-35B-A3B-NVFP4
M8=/models/Qwen3-8B
B=/target/release/intarsia
stage_plans() {  # GPU (TEST_GPU): plan show --json of both trees for the 35B and the 8B, kernel-per-op diff, calls per decode step
  local O="$INTARSIA_REPO/bench-out/plans" t m
  mkdir -p "$O"
  for t in fusion base; do
    for m in 35b 8b; do
      local repo="$INTARSIA_REPO" vol="$INTARSIA_TARGET_VOL" mp=$M35 extra="--mem-budget 29"
      [ $t = base ] && repo="$BASE_REPO" && vol="$BASE_VOL"
      [ $m = 8b ] && mp=$M8 && extra=""
      INTARSIA_REPO="$repo" INTARSIA_TARGET_VOL="$vol" HIP_VISIBLE_DEVICES=${TEST_GPU:-0} \
        $D $B plan show --backend hip --model $mp $extra --max-seq 4096 --json > "$O/$t-$m.json" 2> "$O/$t-$m.err"
      echo "plan $t $m: exit $? $(wc -c < "$O/$t-$m.json") bytes"
    done
  done
  python3 "$INTARSIA_REPO/scripts/hip-decode/plan_diff.py" "$O" 2>&1 | tee "$O/diff.txt"
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
