#!/bin/bash
# DSv4.1 CUDA prefill: three bit-identical experiments in ONE fleetq job on spark2 (next to vLLM). NOT RUN YET (written without nvcc or a GPU).
#   E2  TF-shaped attn_sparse_mma block: 16 heads, 128 threads, Q fragments in registers, __launch_bounds__(128, 2)   (-DATTN_TFBLOCK)
#   E5  interleaved four-row softmax                                                                                      (-DATTN_SM4)
#   E3  rope_tail in place on the 2 * half rope dims only, two forms                                              (-DROPE_INPLACE, -DROPE_INPLACE_SHFL)
# Variants built and compared with the unmodified kernel, alternating 6 pairs of 15 reps:
#   e2 = E2, e2e5 = E2 + E5, e5 = E5 alone (attention, CSA T=37/512/2048, HCA T=512, plus T=11 split path for the hash);  ip, ipshfl = E3 (rope shapes).
# Stop rules: any hash mismatch = negative (not tuning); E2 gain at CSA T=2048 under 12% = negative.
#
# The exact command (see docs/results/2026-11-03-dsv41-attn-tfblock.md), from the repo root of the branch dsv41/attn-tfblock:
#   fleetq run --host spark2 --quiet gpu,cpu --mem 14 --expect 20 --label "dsv41 attn-tfblock: E2/E5/E3 build + A/B (one job)" -- \
#       bash docs/results/2026-11-03-dsv41-attn-tfblock/raw/bench/attn-tfblock-bench.sh
# One job, not a build job plus a bench job: `--quiet gpu,cpu` keeps every other job off the box for the whole run, and the build (cpu, about 4 nvcc)
# is finished before the first timing starts, so the build cannot disturb the timings; it also loads the box no more than the quiet claim already forbids
# others to. nvcc is capped at 4 concurrent jobs (xargs -P4, MAX_JOBS=4) as the Spark memory rule asks.
#
# Modes (all of them stop before touching anything when asked to):
#   (default)   driver, runs on the machine that holds the repo, inside the fleetq slot: copy sources (tar over ssh) to the Spark, run the remote part over ssh -tt,
#               fetch the logs to <this dir>/../run/, remove ~/intarsia-dsv41-tfblock on the Spark (also on failure or interruption).
#   --remote    the part that runs on the Spark (build, calibrate, run, tables, remove build trees).
#   --dry-run   print every step with its exact commands and run nothing: no ssh, no tar to the Spark, no fleetq, no nvcc. Combine with --remote for the Spark side.
#   --selftest  offline: dry-run both patches against kernels/cuda, run calibration + table code on the recorded ab-run-1.log and bench-vec0.txt.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../../.." && pwd)
MEMINFO=${MEMINFO:-/proc/meminfo}   # overridable for the offline plumbing test
MODE=driver; DRY=0; HOST=${HOST:-spark2}; PAIRS=${PAIRS:-6}; ITERS=${ITERS:-20}; REPS=${REPS:-15}; JOBS=${JOBS:-4}; TOL=${TOL:-0.06}
while [ $# -gt 0 ]; do
  case "$1" in
    --remote) MODE=remote;; --dry-run) DRY=1;; --selftest) MODE=selftest;;
    --host) HOST=$2; shift;; --pairs) PAIRS=$2; shift;; --iters) ITERS=$2; shift;; --reps) REPS=$2; shift;;
    -h|--help) sed -n '1,25p' "$0"; exit 0;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
  shift
done

WORKNAME=intarsia-dsv41-tfblock           # new build tree on the Spark (the earlier ~/intarsia-dsv41-attnfuse is deleted)
VARIANTS="e2 e2e5 e5"                      # attention variants
RVARIANTS="ip ipshfl"                      # rope variants
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
# run a command, or in dry-run mode print it
run() { if [ "$DRY" = 1 ]; then printf 'DRY:'; printf ' %q' "$@"; echo; else "$@"; fi; }
# run a shell snippet (pipelines, redirections), or print it
runsh() { if [ "$DRY" = 1 ]; then echo "DRY: $1"; else bash -ec "$1"; fi; }

# ---------------------------------------------------------------------------------------------------------------- selftest
if [ "$MODE" = selftest ]; then
  F="$ROOT/docs/results/2026-11-03-dsv41-attn-fusion/raw/bench"
  T=$(mktemp -d "$ROOT/.worktrees/selftest.XXXXXX" 2>/dev/null || { mkdir -p "$ROOT/.worktrees"; mktemp -d "$ROOT/.worktrees/selftest.XXXXXX"; })
  trap 'rm -rf "$T"' EXIT
  log "patch --dry-run against the current kernels"
  cp "$ROOT/kernels/cuda/dsv41_attn.cu" "$T/a.cu"; cp "$ROOT/kernels/cuda/dsv41_misc.cu" "$T/m.cu"
  patch --dry-run "$T/a.cu" < "$HERE/attn-tfblock.patch"
  patch --dry-run "$T/m.cu" < "$HERE/rope-inplace.patch"
  log "calibration code on the recorded unmodified runs (must PASS)"
  python3 -I "$HERE/tfblock-table.py" calib --ab "$F/ab-run-1.log" --rope "$F/bench-vec0.txt" --expected "$HERE/expected-hashes.txt"
  log "table code on the recorded attn-2blk A/B (must print the 2blk doc's rows and NEGATIVE (stop rule))"
  python3 -I "$HERE/tfblock-table.py" ab --variant var="$F/ab-run-1.log" --blocks var=2 --stop var=12 --expected "$HERE/expected-hashes.txt"
  log "rope table code on the recorded row-vector run (must print MISMATCH / NEGATIVE)"
  python3 -I "$HERE/tfblock-table.py" rope --base "$F/bench-vec0.txt" --variant vec1="$F/bench-vec1.txt" --expected "$HERE/expected-hashes.txt" | tail -4
  exit 0
fi

# ---------------------------------------------------------------------------------------------------------------- remote half
if [ "$MODE" = remote ]; then
  if [ "$DRY" = 1 ]; then W='~/'$WORKNAME; else W=$HOME/$WORKNAME; fi
  SRC=$W/src; B=$W/bench; O=$W/obj; L=$W/logs; MEMPID=
  export PATH=/usr/local/cuda/bin:$PATH MAX_JOBS=$JOBS FLASHINFER_NVCC_THREADS=1
  # build trees go on every exit path; the logs stay for the driver to fetch (the driver removes $W itself)
  cleanup() { local rc=$?; trap - EXIT HUP TERM INT; [ -n "${MEMPID:-}" ] && kill "$MEMPID" 2>/dev/null || true
              if [ "$DRY" != 1 ]; then rm -rf "$O" "$SRC"; log "removed build trees $O and $SRC (logs kept in $L, exit $rc)"; fi; exit $rc; }
  trap cleanup EXIT HUP TERM INT
  log "== remote part on $(hostname), pairs $PAIRS, iters $ITERS, reps $REPS, nvcc jobs $JOBS"
  run mkdir -p "$O" "$L"

  log "== 0. preflight"
  runsh "command -v nvcc cuobjdump python3 patch xargs; nvcc --version | tail -2; nvidia-smi -L"
  runsh "awk '/MemAvailable/ {printf \"MemAvailable %.1f GB\\n\", \$2/1048576}' $MEMINFO; systemctl is-active earlyoom || true"
  # DGX Spark rule: below 10 GB available (15 GB without earlyoom) kill the build rather than let the box swap
  runsh "A=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' $MEMINFO); [ \"\$A\" -ge 20 ] || { echo \"only \$A GB available, refusing to build (need 20)\"; exit 4; }"
  runsh "md5sum $SRC/kernels/cuda/dsv41_attn.cu $SRC/kernels/cuda/dsv41_misc.cu | tee $L/src-md5.txt; echo 'recorded dsv41_attn.cu md5 of the A/B baseline (main 47af18777): b036645eb9f9e9f7edabc4662f4ce2b1; a different md5 may legitimately move the recorded hashes, calibration then decides'"

  log "== 1. sources: unmodified copies, patched copies (flags only enable variants), occupancy probe appended"
  run cp "$SRC/kernels/cuda/dsv41_attn.cu" "$O/attn_base_src.cu"
  run cp "$SRC/kernels/cuda/dsv41_attn.cu" "$O/attn_var_src.cu"
  runsh "patch $O/attn_var_src.cu < $B/attn-tfblock.patch"
  run cp "$SRC/kernels/cuda/dsv41_misc.cu" "$O/misc_var.cu"
  runsh "patch $O/misc_var.cu < $B/rope-inplace.patch"
  runsh "cat $O/attn_base_src.cu $B/occ_probe.inc > $O/attn_base.cu"
  runsh "cat $O/attn_var_src.cu $B/occ_probe.inc > $O/attn_var.cu"
  for f in attn_ab_bench rope_ip_bench; do
    runsh "sed 's#\"../../../../../kernels/abi/ik_abi.h\"#\"abi/ik_abi.h\"#' $B/$f.cu > $O/$f.cu"
  done

  log "== 2. build, at most $JOBS nvcc at a time (flags of crates/intarsia-backend-cuda/build.rs; sm_121)"
  FL="-c -O3 --std=c++17 -lineinfo --expt-relaxed-constexpr -arch=sm_121 -I $SRC/kernels/cuda -I $SRC/kernels"
  BL="-c -O3 --std=c++17 -arch=sm_121 -I $SRC/kernels"
  K=$SRC/kernels/cuda
  # each job is a small script file (portable: no xargs -I, whose replacement limit is 255 bytes on BSD)
  J=$O/jobs
  job() {  # job <kind c|l> <name> <command...>: writes the job script, or prints the command in dry-run
    local kind=$1 name=$2; shift 2
    if [ "$DRY" = 1 ]; then echo "DRY: [$kind-$name] $* > $L/$kind-$name.log 2>&1"; else mkdir -p "$J"; echo "$* > $L/$kind-$name.log 2>&1" > "$J/$kind-$name.sh"; fi
  }
  runjobs() {  # runjobs <kind>: at most $JOBS at a time; on failure show the tails of that kind's logs
    [ "$DRY" = 1 ] && { echo "DRY: ls $J/$1-*.sh | xargs -P$JOBS -n1 bash"; return 0; }
    ls "$J"/$1-*.sh | xargs -P"$JOBS" -n1 bash || { log "$1 step failed; tails of the logs:"; for f in "$L"/$1-*.log; do echo "== $f"; tail -n 15 "$f"; done; exit 5; }
  }
  job c hc nvcc $FL $K/dsv41_hc.cu -o $O/hc.o
  job c hc_sub nvcc $FL $K/dsv41_hc_sub.cu -o $O/hc_sub.o
  job c misc_base nvcc $FL $K/dsv41_misc.cu -o $O/misc_base.o
  job c misc_ip nvcc $FL -DROPE_INPLACE $O/misc_var.cu -o $O/misc_ip.o
  job c misc_ipshfl nvcc $FL -DROPE_INPLACE_SHFL $O/misc_var.cu -o $O/misc_ipshfl.o
  job c attn_base nvcc $FL $O/attn_base.cu -o $O/attn_base.o
  job c attn_e2 nvcc $FL -DATTN_TFBLOCK $O/attn_var.cu -o $O/attn_e2.o
  job c attn_e2e5 nvcc $FL -DATTN_TFBLOCK -DATTN_SM4 $O/attn_var.cu -o $O/attn_e2e5.o
  job c attn_e5 nvcc $FL -DATTN_SM4 $O/attn_var.cu -o $O/attn_e5.o
  job c attn_ab_bench nvcc $BL $O/attn_ab_bench.cu -o $O/attn_ab_bench.o
  job c rope_ip_bench nvcc $BL $O/rope_ip_bench.cu -o $O/rope_ip_bench.o
  if [ "$DRY" != 1 ]; then
    # memory watchdog for the build: kill nvcc when available memory falls under the rule's floor
    FLOOR=10; systemctl is-active --quiet earlyoom 2>/dev/null || FLOOR=15
    ( while sleep 5; do A=$(awk '/MemAvailable/ {print int($2/1048576)}' "$MEMINFO"); echo "$(date +%T) MemAvailable ${A} GB" >> "$L/mem.log"
        if [ "$A" -lt "$FLOOR" ]; then echo "MemAvailable ${A} GB < ${FLOOR}: killing nvcc" | tee -a "$L/mem.log"; pkill -x nvcc || true; pkill -x cicc || true; fi; done ) &
    MEMPID=$!
  fi
  runjobs c
  # links: the proven recipe of ab-build.sh (bench + hc/hc_sub + misc + the attention object); every binary links the same non-attention objects
  LK="-O3 --std=c++17 -arch=sm_121"
  for v in base $VARIANTS; do job l ab_$v nvcc $LK $O/attn_ab_bench.o $O/misc_base.o $O/hc.o $O/hc_sub.o $O/attn_$v.o -o $O/ab_$v; done
  job l rope_base nvcc $LK $O/rope_ip_bench.o $O/misc_base.o $O/hc.o $O/hc_sub.o $O/attn_base.o -o $O/rope_base
  for v in $RVARIANTS; do job l rope_$v nvcc $LK $O/rope_ip_bench.o $O/misc_$v.o $O/hc.o $O/hc_sub.o $O/attn_base.o -o $O/rope_$v; done
  runjobs l
  if [ -n "$MEMPID" ]; then kill "$MEMPID" 2>/dev/null || true; wait "$MEMPID" 2>/dev/null || true; MEMPID=; fi
  log "== 3. resource usage (spills must stay LOCAL 0 for the E2 objects), rope SASS contraction"
  for v in base $VARIANTS; do
    runsh "echo '== attn_$v.o'; cuobjdump -res-usage $O/attn_$v.o 2>&1 | grep -A1 'attn_sparse_mma' | grep -v '^--' | tee -a $L/res-usage.txt"
  done
  for v in base $RVARIANTS; do
    # per rope_tail kernel: FFMA / FMUL / FADD / MUFU counts, and the SASS itself for diagnosing a hash mismatch without another window
    runsh "cuobjdump -sass $O/misc_$v.o 2>/dev/null | awk -v tag=$v -v out=$L/rope-sass-$v.txt -f $B/rope-sass.awk | tee -a $L/rope-sass-summary.txt"
  done

  log "== 4. calibration: the unmodified kernel must reproduce 1.30 ms (CSA T=512), 5.04 ms (CSA T=2048) and the recorded hashes, else stop"
  stamp() { echo "## pair $1 $2 $(date +%T)"; }
  if [ "$DRY" = 1 ]; then
    echo "DRY: { echo '## pair 0 base <time>'; $O/ab_base $ITERS $REPS; } > $L/calib-ab.log ; { echo '## pair 0 base'; $O/rope_base $ITERS 0; } > $L/calib-rope.log"
    echo "DRY: python3 -I $B/tfblock-table.py calib --ab $L/calib-ab.log --rope $L/calib-rope.log --expected $B/expected-hashes.txt --tol $TOL   (exit 3 = stop)"
  else
    { stamp 0 base; "$O/ab_base" "$ITERS" "$REPS"; } > "$L/calib-ab.log"
    { stamp 0 base; "$O/rope_base" "$ITERS" 0; } > "$L/calib-rope.log"
    cat "$L/calib-ab.log" | cut -c1-170
    if ! python3 -I "$B/tfblock-table.py" calib --ab "$L/calib-ab.log" --rope "$L/calib-rope.log" --expected "$B/expected-hashes.txt" --tol "$TOL" | tee "$L/calib.txt"; then
      log "CALIBRATION FAILED: stopping before the variants (the recorded numbers are not reproduced on this box/build)"; exit 3
    fi
  fi

  log "== 5. alternating A/B: $PAIRS pairs, variants interleaved per round, odd pairs base first, even pairs variant first; each process $REPS reps x $ITERS launches"
  for i in $(seq 1 "$PAIRS"); do
    for v in $VARIANTS; do
      if [ $((i % 2)) = 1 ]; then order="base var"; else order="var base"; fi
      for role in $order; do
        bin=$O/ab_$v; [ "$role" = base ] && bin=$O/ab_base
        if [ "$DRY" = 1 ]; then [ "$i" = 1 ] && echo "DRY: pair $i $v: { echo '## pair $i $role <time>'; $bin $ITERS $REPS; } >> $L/ab-$v.log   (order: $order)"
        else { stamp "$i" "$role"; "$bin" "$ITERS" "$REPS"; } >> "$L/ab-$v.log"; fi
      done
    done
    for v in $RVARIANTS; do
      if [ $((i % 2)) = 1 ]; then order="base var"; else order="var base"; fi
      for role in $order; do
        if [ "$role" = base ]; then bin=$O/rope_base; mode=0; out=$L/rope-base.log; else bin=$O/rope_$v; mode=1; out=$L/rope-$v.log; fi
        if [ "$DRY" = 1 ]; then [ "$i" = 1 ] && echo "DRY: pair $i $v: { echo '## pair $i $role <time>'; $bin $ITERS $mode; } >> $out   (order: $order)"
        else { stamp "$i" "$role"; "$bin" "$ITERS" "$mode"; } >> "$out"; fi
      done
    done
  done
  [ "$DRY" = 1 ] && echo "DRY: (the loop repeats for pairs 2..$PAIRS with the order swapped on even pairs; attention variants: $VARIANTS; rope variants: $RVARIANTS)"

  log "== 6. tables and verdicts"
  AB="--variant e2=$L/ab-e2.log --variant e2e5=$L/ab-e2e5.log --variant e5=$L/ab-e5.log --blocks e2=2,e2e5=2,e5=1 --stop e2=12 --stop e2e5=12 --expected $B/expected-hashes.txt"
  RP="--base $L/rope-base.log --variant ip=$L/rope-ip.log --variant ipshfl=$L/rope-ipshfl.log --expected $B/expected-hashes.txt"
  runsh "python3 -I $B/tfblock-table.py ab $AB | tee $L/table-ab.txt"
  runsh "python3 -I $B/tfblock-table.py rope $RP | tee $L/table-rope.txt"
  log "== done (build trees are removed by the exit trap)"
  exit 0
fi

# ---------------------------------------------------------------------------------------------------------------- driver half
OUT=$HERE/../run
# files move as tar over ssh (no rsync needed on the Spark); COPYFILE_DISABLE / --no-xattrs keep macOS metadata out of the archive
tarpush() {  # tarpush <local dir> <remote dir> <names...>
  local from=$1 to=$2; shift 2
  if [ "$DRY" = 1 ]; then echo "DRY: tar -C $from -cf - $* | ssh $HOST 'tar -C $to -xf -'"; else COPYFILE_DISABLE=1 tar -C "$from" --no-xattrs -cf - "$@" | ssh "$HOST" "tar -C $to -xf -"; fi
}
tarfetch() {  # the Spark's logs dir into $OUT
  if [ "$DRY" = 1 ]; then echo "DRY: ssh $HOST 'tar -C ~/$WORKNAME/logs -cf - .' | tar -C $OUT -xf -"; else ssh "$HOST" "tar -C ~/$WORKNAME/logs -cf - ." | tar -C "$OUT" -xf -; fi
}
log "== driver on $(hostname): host $HOST, repo $ROOT, logs to $OUT"
if [ "$DRY" != 1 ] && [ -z "${FLEETQ_JOB:-}" ]; then
  echo "refusing to touch $HOST outside a fleetq slot (FLEETQ_JOB is not set). Run it as:" >&2
  echo "  fleetq run --host $HOST --quiet gpu,cpu --mem 14 --expect 20 --label 'dsv41 attn-tfblock: E2/E5/E3 build + A/B (one job)' -- bash $0" >&2
  exit 2
fi
for f in attn-tfblock.patch rope-inplace.patch occ_probe.inc attn_ab_bench.cu rope_ip_bench.cu tfblock-table.py expected-hashes.txt rope-sass.awk attn-tfblock-bench.sh; do
  [ -f "$HERE/$f" ] || { echo "missing $HERE/$f" >&2; exit 2; }
done
finish() {
  local rc=$?; trap - EXIT HUP TERM INT
  log "== driver cleanup (exit $rc): fetch logs, remove $WORKNAME on $HOST"
  run mkdir -p "$OUT"
  if tarfetch; then
    run ssh "$HOST" "rm -rf ~/$WORKNAME && test ! -e ~/$WORKNAME && echo removed ~/$WORKNAME on $HOST"
  elif [ "$DRY" != 1 ] && ssh "$HOST" "test -e ~/$WORKNAME"; then
    log "logs could not be fetched: ~/$WORKNAME on $HOST is left in place for a manual fetch; remove it afterwards"
  else
    log "nothing to fetch or remove on $HOST"
  fi
  exit $rc
}
trap finish EXIT HUP TERM INT
log "== D1. new build tree on $HOST and the files it needs (kernels/{abi,common,cuda} and this bench dir; nothing else)"
run ssh "$HOST" "rm -rf ~/$WORKNAME && mkdir -p ~/$WORKNAME/src/kernels ~/$WORKNAME/bench ~/$WORKNAME/logs"
tarpush "$ROOT/kernels" "~/$WORKNAME/src/kernels" abi common cuda
tarpush "$HERE" "~/$WORKNAME/bench" .
log "== D2. remote part (ssh -tt so that a dropped connection or the fleetq deadline hangs the remote command up)"
RFLAGS="--remote --pairs $PAIRS --iters $ITERS --reps $REPS"
if [ "$DRY" = 1 ]; then
  echo "DRY: ssh -tt $HOST 'bash ~/$WORKNAME/bench/attn-tfblock-bench.sh $RFLAGS' | tr -d '\\r' | tee <log>"
  echo "---- the remote part, dry run (nothing executes) ----"
  DRY=1 bash "$0" --remote --dry-run --pairs "$PAIRS" --iters "$ITERS" --reps "$REPS"
  echo "---- end of remote dry run ----"
else
  mkdir -p "$OUT"
  ssh -tt "$HOST" "bash ~/$WORKNAME/bench/attn-tfblock-bench.sh $RFLAGS" | tr -d '\r' | tee "$OUT/run.log"
fi
log "== D3. (exit trap) fetch logs to $OUT, remove ~/$WORKNAME on $HOST"
