#!/usr/bin/env bash
# ON tatooine inside one fleetq --load cpu job (lane ink-hip-pf2-tiny): the lane dir holds the case-only tree (6e69bdf3e: the new
# InkKvPrologue admission case, main's kernel); build its cli as intarsia-pf2a; then put the fixed ink_attn.hip in place (a fresh
# mtime, so cargo rebuilds the kernels) and build intarsia-pf2b, the intarsia-models `inkling2` test binary, the hip lib tests and
# the core inkling2 case tests.
set -uo pipefail
L=$HOME/intarsiaml-ink-hip-pf2-tiny
cd "$L"
R=docs/results/2026-11-26-inkling2-hip-pf2-tiny/raw
bash $R/build.sh pf2a || exit $?
cp .scratch/ink_attn.hip.fixed kernels/hip/ink_attn.hip && grep -c "p.dr <= p.e" kernels/hip/ink_attn.hip
bash $R/build.sh pf2b || exit $?
cmp -s .scratch/bin/intarsia-pf2a .scratch/bin/intarsia-pf2b && { echo "pf2a == pf2b: the kernel was not rebuilt"; exit 3; }
export INTARSIA_REPO=$L INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-pf2-tiny INTARSIA_CONTAINER_NAME=ink-hip-pf2-tiny-build
trap 'docker rm -f ink-hip-pf2-tiny-build >/dev/null 2>&1' EXIT
deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-models --test inkling2 --no-run < /dev/null > logs/build-mtest-inkling2.log 2>&1; echo "models inkling2 test build exit $?"
deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-backend-hip --lib < /dev/null 2>&1 | grep -aE "test result|FAILED|panicked" | tail -3
deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-core --lib inkling2 < /dev/null 2>&1 | grep -aE "test result|FAILED|panicked" | tail -3
