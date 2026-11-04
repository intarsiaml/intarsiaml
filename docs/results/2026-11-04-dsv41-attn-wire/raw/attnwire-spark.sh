#!/usr/bin/env bash
# DSv4.1 attn-wire: nvcc build + kernels test --quick + plan show + engine bit identity, in ONE fleetq job on spark2 (never spark1).
#   FLEETQ_OWNER=claude-c8e51313 fleetq run --host spark2 --quiet gpu,cpu --mem 20 --expect 40 \
#     --label "dsv41 attn-wire: nvcc build + kernels test --quick + plan show" -- bash <this script>
# Everything lives in ~/intarsia-dsv41-attnwire on spark2 (removed at the end, logs fetched first to docs/results/2026-11-04-dsv41-attn-wire/raw/).
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
OUT=$REPO/docs/results/2026-11-04-dsv41-attn-wire/raw
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

log "== A1 build CLI (nvcc, MAX_JOBS=4)"
run_to build-cli "INTARSIA_BUILD_STAMP=attnwire cargo build --release --features cuda -p intarsia-cli" || exit 1
abort_check
log "== A2 build the test binaries"
run_to build-tests-runtime "cargo test --release -p intarsia-models --test dsv41_runtime --no-run" || exit 1
run_to build-tests-cuda "cargo test --release -p intarsia-backend-cuda --test dsv41_dpos --test dsv41 --no-run"
RT=$(rsh "cd $W/src && ls -t target/release/deps/dsv41_runtime-* | grep -v '\\.d\$' | head -1")
echo "dsv41_runtime binary: $RT"
rsh "cd $W/src && cp $RT $W/bin-runtime-head"
abort_check

log "== A3 CUDA kernel tests"
TAILN=12 run_to cuda-dpos "INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test dsv41_dpos -- --test-threads 1"
TAILN=12 run_to cuda-dsv41 "INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test dsv41 -- --test-threads 1"
abort_check
log "== A4 kernels test --quick"
TAILN=14 run_to kernels-quick "target/release/intarsia kernels test --backend cuda --quick"
FX=crates/intarsia-models/tests/fixtures/dsv41/mia_headers
for op in attn_sparse rope_tail; do
  TAILN=14 run_to kernels-quick-mia-$op "target/release/intarsia kernels test --backend cuda --quick -m $FX --op $op -v"
done
abort_check

log "== A5 plan show"
FX=crates/intarsia-models/tests/fixtures/dsv41/mia_headers
for v in 1 0; do
  run_to plan-cuda-ip$v "INTARSIA_DSV41_ROPE_INPLACE=$v target/release/intarsia plan show -m $FX --backend cuda --json" >/dev/null
  run_to plan-cuda-ip$v-txt "INTARSIA_DSV41_ROPE_INPLACE=$v target/release/intarsia plan show -m $FX --backend cuda" >/dev/null
done
run_to plan-cpu "target/release/intarsia plan show -m $FX --backend cpu --json" >/dev/null
rsh "cd $W/logs && for v in 1 0; do md5sum plan-cuda-ip\$v.log; done; md5sum plan-cpu.log; diff <(grep -v -iE 'built|stamp|took|ms\\b' plan-cuda-ip0.log) <(grep -v -iE 'built|stamp|took|ms\\b' plan-cuda-ip1.log) | head -30; true"

log "== A6 engine runs on the 3-layer cache (cuda)"
CACHE='$HOME/intarsia-dsv41-tp2/cache/l3-r0of1'
FIXT='$HOME/intarsia-dsv41-tp2/fixtures'
for v in 1 0; do
  TAILN=6 run_to rt-dpos-graphs-ip$v "env INTARSIA_DSV41_ROPE_INPLACE=$v INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=$CACHE INTARSIA_DSV41_DPOS_GRAPHS=1 INTARSIA_DSV41_DP_SPEC=1 INTARSIA_DSV41_DP_PROMPT=140 $W/bin-runtime-head device_position_passes_equal_host_passes --nocapture --test-threads 1"
  TAILN=6 run_to rt-dpos-host-ip$v "env INTARSIA_DSV41_ROPE_INPLACE=$v INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=$CACHE $W/bin-runtime-head device_position_passes_equal_host_passes --nocapture --test-threads 1"
  TAILN=8 run_to rt-layers-ip$v "env INTARSIA_DSV41_ROPE_INPLACE=$v INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=$CACHE INTARSIA_DSV41_FIXTURES=$FIXT $W/bin-runtime-head few_layer_runtime_matches_the_official_layer_outputs --nocapture --test-threads 1"
  abort_check
done
rsh "cd $W/logs && grep -h 'RUNTIME output hash' rt-dpos-*.log | sort | uniq -c"

if [[ -n ${DEADLINE_EPOCH:-} && $(date +%s) -gt $DEADLINE_EPOCH ]]; then log "past the deadline: phase B skipped"; exit 0; fi
log "== B base attention kernel (origin/main's dsv41_attn.cu), rebuild the runtime test binary"
git -C "$REPO" show origin/main:kernels/cuda/dsv41_attn.cu | ssh -o BatchMode=yes $H "cat > $REL/src/kernels/cuda/dsv41_attn.cu"
rsh "touch $W/src/kernels/cuda/dsv41_attn.cu; md5sum $W/src/kernels/cuda/dsv41_attn.cu"
git -C "$REPO" show origin/main:kernels/cuda/dsv41_attn.cu | md5
run_to build-base-attn "cargo test --release -p intarsia-models --test dsv41_runtime --no-run" || exit 1
RT2=$(rsh "cd $W/src && ls -t target/release/deps/dsv41_runtime-* | grep -v '\\.d\$' | head -1")
rsh "cd $W/src && cp $RT2 $W/bin-runtime-baseattn"
abort_check
TAILN=6 run_to rt-dpos-graphs-baseattn-ip0 "env INTARSIA_DSV41_ROPE_INPLACE=0 INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=$CACHE INTARSIA_DSV41_DPOS_GRAPHS=1 INTARSIA_DSV41_DP_SPEC=1 INTARSIA_DSV41_DP_PROMPT=140 $W/bin-runtime-baseattn device_position_passes_equal_host_passes --nocapture --test-threads 1"
TAILN=6 run_to rt-dpos-host-baseattn-ip0 "env INTARSIA_DSV41_ROPE_INPLACE=0 INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=$CACHE $W/bin-runtime-baseattn device_position_passes_equal_host_passes --nocapture --test-threads 1"
TAILN=8 run_to rt-layers-baseattn-ip0 "env INTARSIA_DSV41_ROPE_INPLACE=0 INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=$CACHE INTARSIA_DSV41_FIXTURES=$FIXT $W/bin-runtime-baseattn few_layer_runtime_matches_the_official_layer_outputs --nocapture --test-threads 1"
rsh "cd $W/logs && grep -H 'RUNTIME output hash' rt-dpos-*.log"
log "done"
exit $RC
