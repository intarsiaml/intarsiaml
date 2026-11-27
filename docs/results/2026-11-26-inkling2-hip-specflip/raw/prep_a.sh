#!/usr/bin/env bash
# Mac side, inside `fleetq run --host tatooine-r9700 --load cpu` (lane ink-hip-specflip, option (a)): sync the (a) worktree
# (.scratch/wt-a, branch ink/hip-specflip-b) to tatooine's lane folder, build its hip CLI as .scratch/bin/intarsia-$NAME (+ the
# sampled real-model test exe), then run the cpu tiny tests of the change in the dev image (cpu backend) and core lib.
set -uo pipefail
NAME=${1:?name}
# TESTS: the intarsia-models test targets; LIBS=1 also models lib + core lib; docs_env_vars always
TESTS=${TESTS:-inkling2_dflash2 inkling2_sampled_default recipe_ink_sampled_dflash recipes_doc inkling2_dspark inkling2_dspark_sampled inkling2_dspark_sampled_adapt mtp}; LIBS=${LIBS:-1}
W=/opt/code/intarsiaml/.worktrees/ink-hip-specflip
L=intarsiaml-ink-hip-specflip
R=docs/results/2026-11-26-inkling2-hip-specflip/raw
bash "$W/.scratch/sync_a.sh" || exit 3
( cd "$W/.scratch/wt-a" && echo "$(git rev-parse HEAD) $(git status --porcelain -- crates recipes kernels | wc -l | tr -d ' ') uncommitted" ) > "$W/.scratch/a-commit.txt"
scp -q "$W/.scratch/a-commit.txt" tatooine:$L/.scratch/bin/intarsia-$NAME.commit < /dev/null || exit 4
ssh -o BatchMode=yes tatooine "MTESTS='inkling2_sampled_real' bash ~/$L/$R/build.sh $NAME" < /dev/null || exit 5
ssh -o BatchMode=yes tatooine "TESTS='$TESTS' LIBS=$LIBS; cd ~/$L && export INTARSIA_REPO=\$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-specflip INTARSIA_CONTAINER_NAME=ink-hip-specflip-build; \
  for t in \$TESTS; do \
    deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-models --test \$t -- --test-threads=4 > logs/atest-$NAME-\$t.log 2>&1; echo \"tiny \$t rc \$? | \$(grep -aE '^test result' logs/atest-$NAME-\$t.log | tail -1)\"; grep -aE ' FAILED|panicked' logs/atest-$NAME-\$t.log | head -5; done; \
  [ \$LIBS = 1 ] && deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-models --lib > logs/atest-$NAME-models-lib.log 2>&1; echo \"models lib rc \$? | \$(grep -aE '^test result' logs/atest-$NAME-models-lib.log | tail -1)\"; grep -aE ' FAILED|panicked' logs/atest-$NAME-models-lib.log | head -5; \
  deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-cli --test docs_env_vars > logs/atest-$NAME-docs_env_vars.log 2>&1; echo \"cli docs_env_vars rc \$? | \$(grep -aE '^test result' logs/atest-$NAME-docs_env_vars.log | tail -1)\"; \
  [ \$LIBS = 1 ] && deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-core --lib > logs/atest-$NAME-core.log 2>&1; echo \"core lib rc \$? | \$(grep -aE '^test result' logs/atest-$NAME-core.log | tail -1)\"; grep -aE ' FAILED|panicked' logs/atest-$NAME-core.log | head -5; \
  docker rm -f ink-hip-specflip-build > /dev/null 2>&1" < /dev/null
