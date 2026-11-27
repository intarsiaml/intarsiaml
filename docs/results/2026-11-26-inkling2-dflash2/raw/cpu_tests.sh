#!/usr/bin/env bash
# endor (Mac), inside `fleetq run --host endor --load cpu`: the cpu tests this lane's change touches, release build.
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-dflash2
rc=0
cargo test --release -q -p intarsia-core --lib 2>&1 | grep -aE "^test result|FAILED|panicked" ; [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
cargo test --release -q -p intarsia-models --lib 2>&1 | grep -aE "^test result|FAILED|panicked" ; [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
for t in inkling2_dflash2 dflash2_cpu dflash_cpu inkling2_dspark inkling2_dspark_cut recipes_doc; do
  echo "== intarsia-models --test $t"
  cargo test --release -q -p intarsia-models --test $t 2>&1 | grep -aE "^test result|FAILED|panicked|^test .* FAILED" ; [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
done
echo "== intarsia-cli --test docs_env_vars"
cargo test --release -q --locked -p intarsia-cli --test docs_env_vars 2>&1 | grep -aE "^test result|FAILED|panicked" ; [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
echo "cpu tests exit $rc"
exit $rc
