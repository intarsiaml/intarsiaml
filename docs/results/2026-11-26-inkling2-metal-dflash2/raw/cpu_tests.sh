#!/usr/bin/env bash
# endor (Mac), inside `fleetq run --host endor --load cpu`: cpu tests of the Inkling DFlash2 path + the Metal build (CLI and test binaries, --no-run).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-metal-dflash2
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
rc=0
for t in inkling2_dflash2 dflash2_cpu dflash_cpu inkling2_dspark inkling2_dspark_cut recipes_doc; do
  echo "== intarsia-models --test $t $(date +%T)"
  cargo test --release -q -p intarsia-models --test $t 2>&1 | grep -aE "^test result|FAILED|panicked|^test .* FAILED|^error" ; [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
done
echo "== metal test binaries --no-run $(date +%T)"
cargo test --release -p intarsia-models --test inkling2 --test inkling2_dspark --test inkling2_dspark_cut --test inkling2_verify_softcap --test inkling2_dflash2 --no-run 2>&1 | tail -3
cargo test --release -p intarsia-backend-metal --test dflash2 --no-run 2>&1 | tail -2
cargo test --release -q -p intarsia-core --lib 2>&1 | grep -aE "^test result|FAILED|panicked"; [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
echo "cpu tests exit $rc $(date +%T)"
exit $rc
