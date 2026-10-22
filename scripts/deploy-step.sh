#!/usr/bin/env bash
# The steps scripts/deploy.sh runs ON the deploy host (or inside the HIP dev container), with the tree as the working
# directory. deploy.sh copies this file to <tree>/DEPLOY_STEP.sh; it is not meant to be run by hand.
#   build <backend> <jobs> <build-stamp> [source-date-epoch]   (scripts/repro-build.sh: same sha, same bytes in any directory)
#   smoke <backend> <bin> <kernels-test-args...>
#   probe <bin>                         key=value lines for the stamp (toolchain, lock, binary hash)
#   run   <bin> <outdir> <cmd...>   (words are joined and evaluated by bash)
set -uo pipefail
step=${1:?}; shift
[ -f DEPLOY_ENV.sh ] && . ./DEPLOY_ENV.sh   # knobs recorded by deploy.sh --env-file
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
case "$step" in
  build)
    backend=$1 jobs=$2 stamp=$3 sde=${4:-}
    # repro-build.sh = `cargo build --release --locked -p intarsia-cli` plus path remapping and a fixed SOURCE_DATE_EPOCH;
    # the tree has no .git, so the commit time comes from deploy.sh (4th argument)
    [ -n "$sde" ] && export SOURCE_DATE_EPOCH="$sde"
    if [ -f scripts/repro-build.sh ]; then
      INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS="$jobs" \
        bash scripts/repro-build.sh --features "$backend" 2>&1 | tail -5
    else   # a ref from before repro-build.sh existed (replaying an old bundle)
      INTARSIA_BUILD_STAMP="$stamp" CARGO_BUILD_JOBS="$jobs" \
        cargo build --release --locked --features "$backend" -p intarsia-cli 2>&1 | tail -5
    fi
    exit "${PIPESTATUS[0]}" ;;
  smoke)
    backend=$1 bin=$2; shift 2
    : > DEPLOY_SMOKE.log
    "$bin" doctor --backend "$backend" --json --out DEPLOY_DOCTOR.json >/dev/null; rc=$?
    echo "doctor rc=$rc" | tee -a DEPLOY_SMOKE.log          # 1 = a pitfall check failed; the JSON is still written
    "$bin" kernels test --backend "$backend" "$@" 2>&1 | tee -a DEPLOY_SMOKE.log | tail -15
    krc=${PIPESTATUS[0]}
    echo "kernels-test rc=$krc" | tee -a DEPLOY_SMOKE.log
    [ -s DEPLOY_DOCTOR.json ] && [ "$krc" = 0 ] ;;
  probe)
    bin=$1
    echo "rustc=$(rustc -V)"; echo "cargo=$(cargo -V)"
    [ -f rust-toolchain.toml ] && echo "toolchain_file=$(sha rust-toolchain.toml | cut -c1-16)"
    echo "lock_sha256=$(sha Cargo.lock)"
    if [ -x "$bin" ]; then echo "binary_sha256=$(sha "$bin")"; echo "binary_bytes=$(wc -c < "$bin" | tr -d ' ')"; fi
    command -v nvcc >/dev/null && echo "nvcc=$(nvcc --version | grep release | sed 's/^ *//')"
    command -v hipcc >/dev/null && echo "hipcc=$(hipcc --version 2>/dev/null | head -1)"
    [ -r /opt/rocm/.info/version ] && echo "rocm=$(cat /opt/rocm/.info/version)"
    true ;;
  run)
    bin=$1 out=$2; shift 2
    mkdir -p "$out"
    export INTARSIA_BIN="$bin" INTARSIA_REPLAY_OUT="$out"
    eval "$*" ;;   # a shell string: $INTARSIA_BIN and $INTARSIA_REPLAY_OUT expand here
  *) echo "deploy-step: unknown step $step" >&2; exit 2 ;;
esac
