#!/usr/bin/env bash
# P2' MoeBlock on tatooine (docs/results/2026-10-19-p2prime-moeblock.md). Runs ON tatooine inside a fleetq lease.
#   CPU-only slot (--mem 0, radiance stays up):  p2prime-moeblock.sh mutbuild     (the p2p-* mutant test binaries)
#   GPU window (fusion-window.sh, FUSION_SCRIPT=p2prime-moeblock.sh): bench, t:<test>[:<filter>], mutrun, adm, plans, env:..., r:<remote.sh stage>
# Tree INTARSIA_REPO (default ~/intarsiaml-p2p), cargo volume intarsia-cargo-target-p2s0. The A/B arms are the SAME binary
# with different env (default / INTARSIA_HIP_MOE_BLOCK=1 / INTARSIA_HIP_MOE_V3=1), so no base tree is needed:
#   ab:<arm>:<tag>    arm = def | blk | v3: remote.sh pf35:<arm>-<tag> (or pf8x with ab8:) with that env in the serve container
#   ids               256 greedy ids after the 4408-token prompt, default vs INTARSIA_HIP_MOE_BLOCK=1 (remote.sh ids4k, same tree)
#   env:NAME=VALUE / unenv:NAME   export / drop an env var for the later stages of this lane (INTARSIA_HIP_* reach the container)
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-p2p}"
export INTARSIA_TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-p2s0}"
cd "$INTARSIA_REPO"
D=deploy/tatooine/dev.sh
P2PMUT="${P2PMUT:-p2p-gu-nofence p2p-sh-nofence p2p-nofence p2p-norefresh p2p-noorder p2p-ticket p2p-wait-short p2p-wait-routed p2p-swap p2p-noreset p2p-topk-rank p2p-sdc-noround p2p-topk-wround}"
ts() { date +%T; }

stage_bench() {  # GPU: the M1 microbench (tests/moe_block.rs moe_block_timing)
  HIP_VISIBLE_DEVICES=${TEST_GPU:-0} INTARSIA_HIP_MOE_BLOCK_BENCH=1 timeout 300 $D cargo test --release -p intarsia-backend-hip --test moe_block \
    moe_block_timing -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|panicked|moe_block_timing|^  " | cut -c1-200 | tail -20
}

stage_t() {  # t:<test file>[:<filter>]: one hip test binary, single-threaded, with output
  local f="${1%%:*}" filt=""
  [[ "$1" == *:* ]] && filt="${1#*:}"
  HIP_VISIBLE_DEVICES=${TEST_GPU:-0} timeout ${T_TIMEOUT:-600} $D cargo test --release -p intarsia-backend-hip --test "$f" -- $filt --nocapture --test-threads=1 2>&1 \
    | grep -aE "^test |test result|FAILED|panicked|differ|cases bitwise|launches|^  \[|bitwise" | cut -c1-260 | tail -${T_TAIL:-60}
}

stage_mutbuild() {  # CPU only: each mutant's test binary into /target/p2pmut/<name>
  local MUT="${INTARSIA_MUT_DIR:-$HOME/intarsiaml-p2p-mut}" m t exe
  for m in ${MUTATIONS:-$P2PMUT}; do
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
    INTARSIA_REPO="$MUT" $D bash -c "mkdir -p /target/p2pmut && cp '$exe' /target/p2pmut/$m"
    echo "$m: test $t built [$(ts)]"
  done
  rm -rf "$MUT"
  find "$INTARSIA_REPO/kernels" "$INTARSIA_REPO/crates" -type f -exec touch {} +
  echo "NOTE: rebuild this tree (scripts/hip-decode/build.sh) before running binaries from its target volume"
}

stage_isa() {  # CPU only: gfx1201 ISA of moe_block.hip, base and every mutant, the moe_block_kernel<4> body, and the diffs
  local I="$INTARSIA_REPO/bench-out/p2p-isa" m
  rm -rf "$I"; mkdir -p "$I"
  for m in base ${MUTATIONS:-$P2PMUT}; do
    local src="$INTARSIA_REPO/bench-out/p2p-isa-src/$m"
    rm -rf "$src"; mkdir -p "$src"
    cp -r "$INTARSIA_REPO/kernels" "$src/"
    if [ "$m" != base ]; then python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" "$src" "$m" >/dev/null || { echo "$m: DID NOT APPLY"; continue; }; fi
    $D bash -c "cd /work && /opt/rocm/bin/hipcc -std=c++17 -O3 --offload-arch=gfx1201 --cuda-device-only -S bench-out/p2p-isa-src/$m/kernels/hip/moe_block.hip -o bench-out/p2p-isa/$m.s" > "$I/$m.hipcc.txt" 2>&1 || { echo "$m: hipcc FAILED"; continue; }
    awk '$0 ~ /^_Z[^ ]*moe_block_kernelILi4E[^ ]*:/ {on=1} on {print} on && /^\.Lfunc_end/ {exit}' "$I/$m.s" | sed -E 's/\.LBB[0-9]+_/.LBB_/g' > "$I/$m.k4.s"
  done
  rm -rf "$INTARSIA_REPO/bench-out/p2p-isa-src"
  for m in ${MUTATIONS:-$P2PMUT}; do
    [ -s "$I/$m.k4.s" ] || continue
    diff "$I/base.k4.s" "$I/$m.k4.s" > "$I/$m.k4.diff"
    echo "=== $m vs base <4>: $(grep -c '^[<>]' "$I/$m.k4.diff") diff lines ($(wc -l < "$I/$m.k4.s") lines)"
  done
  echo "=== base <4>: global_inv / s_barrier / s_sleep / DEV-scope / atomics"
  grep -nE 'global_inv|s_barrier|s_sleep|scope:SCOPE_DEV|global_wb|global_atomic' "$I/base.k4.s" | head -60
}

stage_mutrun() {  # GPU: every mutant binary must FAIL (or hang -> killed by the timeout)
  local m
  for m in ${MUTATIONS:-$P2PMUT}; do
    printf '%-18s ' "$m"
    HIP_VISIBLE_DEVICES=${MUT_GPU:-0} $D bash -c "timeout -s KILL ${MUT_TIMEOUT:-150} /target/p2pmut/$m ${RUN_FILTER:-} --test-threads=1 --nocapture; echo exit=\$?" 2>&1 \
      | grep -aE "^test result|panicked at|^exit=|differ from the chain, |launches, " | head -5 | cut -c1-200 | tr '\n' ' '
    echo
  done
}

M35=/models/Qwen3.6-35B-A3B-NVFP4
M8=/models/Qwen3-8B
B=/target/release/intarsia
stage_plans() {  # GPU (TEST_GPU): plan show --json default vs INTARSIA_HIP_MOE_BLOCK=1, 35B and 8B, kernel-per-op diff
  local O="$INTARSIA_REPO/bench-out/plans" t m
  mkdir -p "$O"
  for t in def blk; do
    for m in 35b 8b; do
      local mp=$M35 extra="--mem-budget 29"
      [ $m = 8b ] && mp=$M8 && extra=""
      ( [ $t = blk ] && export INTARSIA_HIP_MOE_BLOCK=1
        HIP_VISIBLE_DEVICES=${TEST_GPU:-0} $D $B plan show --backend hip --model $mp $extra --max-seq 4096 --json > "$O/$t-$m.json" 2> "$O/$t-$m.err" )
      echo "plan $t $m: exit $? $(wc -c < "$O/$t-$m.json") bytes"
    done
  done
  python3 "$INTARSIA_REPO/scripts/hip-decode/plan_diff.py" "$O" blk def 2>&1 | tee "$O/diff.txt"
}

stage_adm() {  # GPU (TEST_GPU): FULL admission sweeps (no --quick), 35B and 8B shapes, default and INTARSIA_HIP_MOE_BLOCK=1
  local O="$INTARSIA_REPO/bench-out/adm" t m
  mkdir -p "$O"
  for t in def blk; do
    for m in 35b 8b; do
      local mp=$M35
      [ $m = 8b ] && mp=$M8
      ( [ $t = blk ] && export INTARSIA_HIP_MOE_BLOCK=1
        HIP_VISIBLE_DEVICES=${TEST_GPU:-0} $D $B kernels test --backend hip --model $mp > "$O/$t-$m.log" 2>&1 )
      echo "adm $t $m exit $?: $(grep -a ' kernels, ' "$O/$t-$m.log" | tail -1) | FAIL/REJECT lines: $(grep -acE 'FAIL|REJECT' "$O/$t-$m.log")"
    done
  done
  grep -aE "moe_block" "$O/blk-35b.log" | cut -c1-200 | head -12
}

stage_ids() {  # 256 greedy ids after the 4408-token prompt: INTARSIA_HIP_MOE_BLOCK=1 ("new") vs default ("base"), same tree
  BASE_REPO="$INTARSIA_REPO" BASE_VOL="$INTARSIA_TARGET_VOL" IDS_NEW_ENV="INTARSIA_HIP_MOE_BLOCK=1" \
    bash "$INTARSIA_REPO/scripts/hip-decode/remote.sh" ids4k
}

arm_env() {
  case "$1" in
    def) echo "" ;;
    blk) echo "-e INTARSIA_HIP_MOE_BLOCK=1" ;;
    v3) echo "-e INTARSIA_HIP_MOE_V3=1" ;;
    *) echo "BAD-ARM" ;;
  esac
}

for s in "$@"; do
  echo "################ stage $s start $(ts)"
  if [[ "$s" == r:* ]]; then
    bash "$INTARSIA_REPO/scripts/hip-decode/remote.sh" "${s#r:}"
  elif [[ "$s" == ab:* || "$s" == ab8:* ]]; then  # ab:<arm>:<tag> / ab8:<arm>:<tag>
    IFS=: read -r kind arm tag <<< "$s"
    st=pf35; [ "$kind" = ab8 ] && st=pf8x
    PF_ENV="$(arm_env "$arm")" AT_MOST=1 bash "$INTARSIA_REPO/scripts/hip-decode/remote.sh" "$st:$arm-$tag"
  elif [[ "$s" == env:* ]]; then
    kv="${s#env:}"; export "${kv?}"; echo "export $kv"
  elif [[ "$s" == unenv:* ]]; then
    unset "${s#unenv:}"; echo "unset ${s#unenv:}"
  elif [[ "$s" == filter:* ]]; then
    export RUN_FILTER="${s#filter:}"; echo "RUN_FILTER=$RUN_FILTER"
  elif [[ "$s" == mut:* ]]; then  # mut:<name>:<name>...: mutrun for these mutants only (one lane's share)
    m="${s#mut:}"; MUTATIONS="${m//:/ }" stage_mutrun
  elif [[ "$s" == t:* ]]; then
    stage_t "${s#t:}"
  else
    "stage_$s"
  fi
  echo "################ stage $s end $(ts)"
done
