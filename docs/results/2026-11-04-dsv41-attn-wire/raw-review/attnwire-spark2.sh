#!/usr/bin/env bash
# DSv4.1 attn-wire: nvcc build + kernels test --quick + plan show + engine bit identity, in ONE fleetq job on spark2 (never spark1).
#   FLEETQ_OWNER=claude-c8e51313 fleetq run --host spark2 --quiet gpu,cpu --mem 20 --expect 40 \
#     --label "dsv41 attn-wire: nvcc build + kernels test --quick + plan show" -- bash <this script>
# Everything lives in ~/intarsia-dsv41-attnwire on spark2 (removed at the end, logs fetched first to docs/results/2026-11-04-dsv41-attn-wire/raw-review/).
# Phases, in priority order; phase B only starts when time allows (DEADLINE_EPOCH, default none).
#   A  build the head tree, run: backend-cuda tests dsv41_dpos (in-place rope == copy rope, bitwise) + dsv41 (kernels vs cpu ref),
#      `kernels test --backend cuda --quick`, `plan show` of the Mia header fixture with ROPE_INPLACE=1 and =0 (and the cpu backend),
#      dsv41_runtime on the 3-layer cache: device_position_passes_equal_host_passes (graphs, spec) and the official-layer-output test,
#      each with INTARSIA_DSV41_ROPE_INPLACE=1 and 0 (E3 at engine level: the output hash must be equal)
#   B  swap kernels/cuda/dsv41_attn.cu for origin/main's, rebuild the dsv41_runtime binary, run the device_position test with
#      ROPE_INPLACE=0: its output hash against phase A's ROPE_INPLACE=0 hash is E5 at engine level.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
H=spark2
REL=intarsia-dsv41-attnwire
W='$HOME/'$REL
OUT=$REPO/docs/results/2026-11-04-dsv41-attn-wire/raw-review
mkdir -p "$OUT"
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4 FLASHINFER_NVCC_THREADS=1;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RP export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
RC=0
cleanup() {
  trap - EXIT HUP TERM INT
  log "cleanup: stop memwatch, fetch logs, remove $REL"
  rsh "pkill -f 'intarsia-dsv41-attnwire/memwatch[.]sh'; $W/killmine.sh" >/dev/null 2>&1 || true
  rsync -a "$H:$REL/logs/" "$OUT/" 2>&1 | tail -2 || true
  # the one recursive delete: a real directory directly under the home directory with exactly this name
  rsh "d=$W; case \"\$d\" in \"\$HOME\"/intarsia-dsv41-attnwire) ;; *) echo refusing \$d; exit 3;; esac; [ ! -L \"\$d\" ] && rm -rf -- \"\$d\" && echo removed \$d" || true
  exit $RC
}
trap cleanup EXIT HUP TERM INT

log "== preflight"
rsh "mkdir -p $W/logs $W/src; systemctl is-active earlyoom; awk '/MemAvailable/ {printf \"MemAvailable %.1f GB\\n\", \$2/1048576}' /proc/meminfo; nvidia-smi -L; nvcc --version | tail -1"
# helper scripts (local copies in .scratch, copied into the job directory): memory watch and a kill limited to this job's own processes
scp -q -o BatchMode=yes "$REPO/.scratch/killmine.sh" "$REPO/.scratch/memwatch.sh" "$H:$REL/"
ssh -n -o BatchMode=yes $H "nohup $REL/memwatch.sh > /dev/null 2>&1 &"

log "== seed target and copy the tree"
rsh "test -d $W/src/target || cp -a ~/intarsia-dsv41-tp2/src/target $W/src/target"
rsync -a --no-t --checksum --delete --exclude target --exclude 'target-*' --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .scratch --exclude .worktrees "$REPO/" "$H:$REL/src/"
echo "head $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -- crates kernels | wc -l) files" | tee "$OUT/stamp.txt"
rsh "find $W/src/crates $W/src/kernels -type f -exec touch {} +"

run_to() { # run_to LOGNAME CMD... : run on the Spark in src, log there, print the tail here, set RC on failure
  local name=$1; shift
  log "-- $name"
  rsh "cd $W/src && ( $* ) > $W/logs/$name.log 2>&1; rc=\$?; tail -${TAILN:-4} $W/logs/$name.log; echo \"[$name exit \$rc]\"; exit \$rc" || { RC=1; return 1; }
}
abort_check() { rsh "test ! -e $W/ABORT" || { log "ABORT flag: memory dropped under 10 GB"; RC=2; exit 2; }; }


CACHE='$HOME/intarsia-dsv41-tp2/cache/l3-r0of1'
log "== fast type check (CI cuda cfg: no nvcc), fail early"
TAILN=6 run_to typecheck "INTARSIA_CUDA_CHECK_ONLY=1 CARGO_TARGET_DIR=$W/chk cargo check -p intarsia-backend-cuda --all-targets && INTARSIA_CUDA_CHECK_ONLY=1 CARGO_TARGET_DIR=$W/chk cargo check -p intarsia-models --tests" || exit 1
rsh "rm -rf $W/chk"
log "== build the test binaries (nvcc, MAX_JOBS=4)"
run_to build-tests-cuda "cargo test --release -p intarsia-backend-cuda --test dsv41_dpos --test dsv41 --no-run" || exit 1
run_to build-tests-runtime "cargo test --release -p intarsia-models --test dsv41_runtime --no-run" || exit 1
abort_check
RT=$(rsh "cd $W/src && ls -t target/release/deps/dsv41_runtime-* | grep -v '\\.d\$' | head -1")
rsh "cd $W/src && cp $RT $W/bin-runtime-head"
log "== kernel tests: in-place path counter, refusal, E5 bit guard"
TAILN=14 run_to cuda-dpos "INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test dsv41_dpos -- --test-threads 1 --nocapture"
TAILN=14 run_to cuda-dsv41 "INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test dsv41 attn_sparse -- --test-threads 1 --nocapture"
log "== negative control: the SIMT attention kernel must FAIL the E5 guard"
RC0=$RC
TAILN=14 run_to cuda-guard-simt-control "INTARSIA_REQUIRE_GPU=1 INTARSIA_DSV41_ATTN_SIMT=1 cargo test --release -p intarsia-backend-cuda --test dsv41 attn_sparse_mma_is_bitwise -- --test-threads 1 --nocapture"
rsh "grep -c FAIL $W/logs/cuda-guard-simt-control.log"; RC=$RC0
abort_check
log "== engine: in-place == out-of-place inside the test (prefill logits included)"
TAILN=8 run_to rt-default "env INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=$CACHE INTARSIA_DSV41_DPOS_GRAPHS=1 INTARSIA_DSV41_DP_SPEC=1 INTARSIA_DSV41_DP_PROMPT=140 $W/bin-runtime-head device_position_passes_equal_host_passes --nocapture --test-threads 1"
TAILN=8 run_to rt-long "env INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=$CACHE INTARSIA_DSV41_DPOS_GRAPHS=1 INTARSIA_DSV41_DP_CHUNK=2048 INTARSIA_DSV41_DP_PROMPT=4200 INTARSIA_DSV41_DP_CAP=8192 INTARSIA_DSV41_DP_TOKENS=24 $W/bin-runtime-head device_position_passes_equal_host_passes --nocapture --test-threads 1"
abort_check
if ! rsh "grep -q 'test result: ok' $W/logs/cuda-dsv41.log"; then
  log "== guard failed on the head: run it against main's attention kernel to tell a wrong generator from a changed kernel"
  git -C "$REPO" show origin/main:kernels/cuda/dsv41_attn.cu | ssh -o BatchMode=yes $H "cat > $REL/src/kernels/cuda/dsv41_attn.cu"
  TAILN=14 run_to cuda-guard-base-attn "INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test dsv41 attn_sparse_mma_is_bitwise -- --test-threads 1 --nocapture"
fi
log "done"
exit $RC
