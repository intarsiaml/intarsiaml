#!/usr/bin/env bash
# endor, --load cpu: the cpu tests on the merged head (#544 in) + the Metal-head refusal
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-metal-dflash2
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
rc=0
echo "head $(git rev-parse --short HEAD) $(git status --short | wc -l) dirty"
for t in inkling2_dflash2 dflash2_cpu dflash_cpu inkling2_dspark inkling2_dspark_cut recipes_doc; do
  echo "== intarsia-models --test $t $(date +%T)"
  cargo test --release -q -p intarsia-models --test $t 2>&1 | grep -aE "^test result|FAILED|panicked|^test .* FAILED|^error" ; [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
done
echo "== core lib $(date +%T)"
cargo test --release -q -p intarsia-core --lib 2>&1 | grep -aE "^test result|FAILED|panicked"; [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
echo "== docs_env_vars $(date +%T)"
cargo test --release -q --locked -p intarsia-cli --test docs_env_vars 2>&1 | grep -aE "^test result|FAILED|panicked" ; [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -1
cp target/release/intarsia .scratch/bin/intarsia-br3 && md5 .scratch/bin/intarsia-br3
cargo test --release -p intarsia-models --test inkling2_dflash2 --test inkling2 --test inkling2_dspark --test inkling2_verify_softcap --no-run 2>&1 | grep -E "error|Finished"
echo "cpu tests exit $rc $(date +%T)"
exit $rc
