#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job: (1) save the volume's last build (main2 = origin/main, built from .scratch/main-src) as
# .scratch/bin/intarsia-main2; (2) build the branch as intarsia-$NAME with the gate test targets; (3) the cpu tests the k change
# touches: intarsia-models lib test dspark_default_only_where_the_recipe_sets_it, intarsia-cli docs_env_vars.
set -uo pipefail
LANE="$HOME/intarsiaml-ink-hip-verify"
cd "$LANE"; mkdir -p .scratch/bin logs
DEV="$LANE/deploy/tatooine/dev.sh"
export INTARSIA_REPO=$LANE INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-verify INTARSIA_CONTAINER_NAME=ink-hip-verify-build
if [ -n "${SAVE_MAIN2:-}" ]; then
  $DEV cp /target/release/intarsia /work/.scratch/bin/intarsia-main2 < /dev/null && md5sum .scratch/bin/intarsia-main2
fi
docker rm -f ink-hip-verify-build >/dev/null 2>&1
HTESTS="ink_moe" MTESTS="verify_rows_real inkling2 inkling2_verify_softcap inkling2_dspark inkling2_dspark_cut" bash .scratch/build.sh "${NAME:-verify1}" || exit $?
trap 'docker rm -f ink-hip-verify-build >/dev/null 2>&1' EXIT
$DEV cargo test --release -j8 -p intarsia-models --lib dspark_default < /dev/null > logs/test-models-lib-dspark.log 2>&1
echo "models lib dspark_default exit $?"; grep -aE "^test |test result" logs/test-models-lib-dspark.log | head -8
$DEV cargo test --release -j8 -p intarsia-cli --test docs_env_vars < /dev/null > logs/test-docs-env.log 2>&1
echo "cli docs_env_vars exit $?"; grep -aE "^test |test result|panicked" logs/test-docs-env.log | head -8
