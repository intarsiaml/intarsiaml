#!/usr/bin/env bash
# P2' step 0 on tatooine (docs/results/2026-10-19-p2prime-step0.md): the v3 consumer-ordering mutants against the sentinel
# stress tests (tests/moe_v3.rs). Runs ON tatooine inside a fleetq lease. Like fusion-p2.sh it never touches a container it
# did not start, so it is safe in a CPU slot.
#   CPU-only slot (--mem 0, radiance stays up):   p2prime-step0.sh isa build
#   GPU window (--takes-down radiance, fusion-window.sh with FUSION_SCRIPT=p2prime-step0.sh):  p2prime-step0.sh run
# Tree INTARSIA_REPO (default ~/intarsiaml-p2s0), cargo volume intarsia-cargo-target-p2s0. Binaries: /target/p2s0mut/<name>
# (base = the unmutated moe_v3 test binary, built first and copied out like the mutants).
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-p2s0}"
export INTARSIA_TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-p2s0}"
cd "$INTARSIA_REPO"
D=deploy/tatooine/dev.sh
V3MUT="v3-nofence v3-noorder v3-nobaracq v3-nospinacq v3-fenceonly"
OUT="$INTARSIA_REPO/bench-out/p2s0"
mkdir -p "$OUT"
ts() { date +%T; }

test_exe() {  # the moe_v3 test binary of tree $1 (built if needed)
  INTARSIA_REPO="$1" $D cargo test --release -p intarsia-backend-hip --test moe_v3 --no-run --message-format=json 2>/dev/null \
    | python3 -c 'import json,sys
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    if d.get("reason") == "compiler-artifact" and d.get("executable") and d["target"]["kind"] == ["test"]: print(d["executable"])' | tail -1
}

stage_isa() {  # CPU only: gfx1201 assembly of moe.hip for the base and every v3 mutant; the v3 kernels' bodies and their diffs
  local I="$OUT/isa" m src p f
  rm -rf "$I"; mkdir -p "$I"
  for m in base $V3MUT; do
    src="$OUT/isa-src/$m"
    rm -rf "$src"; mkdir -p "$src"
    cp -r "$INTARSIA_REPO/kernels" "$src/"
    if [ "$m" != base ]; then python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" "$src" "$m" || { echo "$m: DID NOT APPLY"; continue; }; fi
    $D bash -c "cd /work && /opt/rocm/bin/hipcc -S -O3 -std=c++17 --offload-arch=gfx1201 --cuda-device-only \
      -I bench-out/p2s0/isa-src/$m/kernels/hip -o bench-out/p2s0/isa/$m.s bench-out/p2s0/isa-src/$m/kernels/hip/moe.hip" \
      > "$I/$m.hipcc.txt" 2>&1 || { echo "$m: hipcc FAILED"; cat "$I/$m.hipcc.txt" | head; continue; }
    rm -rf "$src"
  done
  rmdir "$OUT/isa-src" 2>/dev/null
  stage_isadiff
}

stage_isadiff() {  # the v3 kernels' bodies out of $OUT/isa/<name>.s, their diffs, and every invalidate / barrier / wait in them
  local I="$OUT/isa" m p f
  for m in base $V3MUT; do
    for p in 1 2 4; do  # one function body per PRE instantiation (the default PRE is 4: moe_pre()); block labels renumbered
      awk -v pre="$p" '$0 ~ ("^_Z[^ ]*moe_experts_p1_v3_kernelILi" pre "E[^ ]*: ") {on=1} on {print} on && /^\.Lfunc_end/ {exit}' \
        "$I/$m.s" | sed -E 's/\.LBB[0-9]+_/.LBB_/g' > "$I/$m.k$p.s"
    done
    echo "$m: $(wc -l < "$I/$m.k4.s") lines in <4>; global_inv / barrier lines: $(grep -cE 'global_inv|s_barrier' "$I/$m.k4.s")"
  done
  for m in $V3MUT; do
    for p in 1 2 4; do diff "$I/base.k$p.s" "$I/$m.k$p.s" > "$I/$m.k$p.diff"; done
    echo "=== $m vs base, <4> (diff lines: <1> $(grep -c '^[<>]' "$I/$m.k1.diff"), <2> $(grep -c '^[<>]' "$I/$m.k2.diff"), <4> $(grep -c '^[<>]' "$I/$m.k4.diff"))"
    cat "$I/$m.k4.diff"
  done
  # every cache invalidate, barrier and wait in the <4> bodies, with line numbers
  for f in base $V3MUT; do
    echo "=== $f <4>: global_inv / s_barrier / s_wait_loadcnt / s_sleep / DEV-scope loads"
    grep -nE 'global_inv|s_barrier|s_sleep|scope:SCOPE_DEV|global_wb|global_atomic' "$I/$f.k4.s"
  done
}

stage_build() {  # CPU only: the base moe_v3 binary, then each mutant's, into /target/p2s0mut/<name>
  local MUT="${INTARSIA_MUT_DIR:-$HOME/intarsiaml-p2s0-mut}" m exe
  exe=$(test_exe "$INTARSIA_REPO")
  if [ -z "$exe" ]; then echo "base: BUILD FAILED"; INTARSIA_REPO="$INTARSIA_REPO" $D cargo test --release -p intarsia-backend-hip --test moe_v3 --no-run 2>&1 | grep -E "^error|-->" | head -20; return 1; fi
  $D bash -c "mkdir -p /target/p2s0mut && cp '$exe' /target/p2s0mut/base"
  echo "base: built [$(ts)]"
  for m in ${MUTATIONS:-$V3MUT}; do
    rm -rf "$MUT"; mkdir -p "$MUT"
    rsync -a --exclude target --exclude bench-out --exclude prof-out "$INTARSIA_REPO/" "$MUT/"
    python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" "$MUT" "$m" || continue
    exe=$(test_exe "$MUT")
    if [ -z "$exe" ]; then echo "$m: BUILD FAILED"; continue; fi
    INTARSIA_REPO="$MUT" $D bash -c "cp '$exe' /target/p2s0mut/$m"
    echo "$m: built [$(ts)]"
  done
  rm -rf "$MUT"
  find "$INTARSIA_REPO/kernels" "$INTARSIA_REPO/crates" -type f -exec touch {} +
  $D bash -c "ls -la /target/p2s0mut/"
}

stage_run() {  # GPU: every binary in RUNS (default base + the mutants), REPS full moe_v3 runs each
  local m r
  for m in ${RUNS:-base $V3MUT}; do
    for r in $(seq 1 "${REPS:-1}"); do
      echo "--- $m run $r [$(ts)] GPU ${MUT_GPU:-0}"
      HIP_VISIBLE_DEVICES=${MUT_GPU:-0} $D bash -c "timeout -s KILL ${MUT_TIMEOUT:-240} /target/p2s0mut/$m ${RUN_FILTER:-} --test-threads=1 --nocapture; echo exit=\$?" 2>&1 \
        | grep -aE "^test |^test result|panicked at|^exit=|launches, |^  \[|differs from v2|read h early" | cut -c1-260 | head -60
    done
  done
}

for s in "$@"; do
  echo "################ stage $s start $(ts)"
  if [[ "$s" == run:* ]]; then  # run:<reps>:<name>:<name>...: REPS and RUNS for one lane of fusion-window.sh
    IFS=: read -r _ reps names <<< "$s"
    REPS="$reps" RUNS="${names//:/ }" stage_run
  elif [[ "$s" == pre:* ]]; then  # pre:<1|2|4>: INTARSIA_HIP_MOE_PRE (the v3 template instance) for the later stages of this lane
    export INTARSIA_HIP_MOE_PRE="${s#pre:}"
    echo "INTARSIA_HIP_MOE_PRE=$INTARSIA_HIP_MOE_PRE"
  elif [[ "$s" == filter:* ]]; then  # filter:<test name>: only that moe_v3 test in the later run stages (P2' M0)
    export RUN_FILTER="${s#filter:}"
    echo "RUN_FILTER=$RUN_FILTER"
  else
    "stage_$s"
  fi
  echo "################ stage $s end $(ts)"
done
