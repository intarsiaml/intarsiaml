#!/usr/bin/env bash
# Mac side, inside `fleetq run --host tatooine-r9700 --load cpu` (lane ink-hip-specflip): PR B candidate on tatooine while the Mac worktree
# stays on PR A (its cpu suite runs there): sync the worktree (PR A head), apply raw/prb_patch.py + prb_patch2.py there (ADAPT = the D1 table),
# build the hip CLI as .scratch/bin/intarsia-$NAME + test exes, run the cpu tiny tests of PR B (cpu backend, in the dev image) and print md5s
# of the patched files (to compare with the same patch applied on the Mac later).
set -uo pipefail
NAME=${1:?name}; ADAPT=${2:?adapt string}
W=/opt/code/intarsiaml/.worktrees/ink-hip-specflip
L=intarsiaml-ink-hip-specflip
R=docs/results/2026-11-26-inkling2-hip-specflip/raw
cd "$W"
bash "$W/$R/sync.sh" || exit 3
ssh -o BatchMode=yes tatooine "cd ~/$L && PRB_ROOT=\$HOME/$L python3 $R/prb_patch.py '$ADAPT' && PRB_ROOT=\$HOME/$L python3 $R/prb_patch2.py '$ADAPT' && echo patched && \
  md5sum crates/intarsia-core/src/recipe.rs crates/intarsia-core/src/model/mtp.rs crates/intarsia-models/src/load.rs crates/intarsia-models/tests/mtp.rs \
    crates/intarsia-models/tests/inkling2_dflash2.rs recipes/inkling2.toml docs/recipes.md docs/tuning.md && \
  echo '$(git rev-parse HEAD) + prb_patch $ADAPT' > .scratch/bin/intarsia-$NAME.commit" < /dev/null || exit 4
ssh -o BatchMode=yes tatooine "MTESTS='inkling2_sampled_real inkling2_dflash2 recipe_ink_sampled_dflash recipes_doc inkling2_sampled_default' bash ~/$L/$R/build.sh $NAME" < /dev/null || exit 5
ssh -o BatchMode=yes tatooine "cd ~/$L && export INTARSIA_REPO=\$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-specflip INTARSIA_CONTAINER_NAME=ink-hip-specflip-build; \
  for t in inkling2_dflash2 recipe_ink_sampled_dflash recipes_doc inkling2_sampled_default; do \
    deploy/tatooine/dev.sh /work/.scratch/bin/t-$NAME-\$t --test-threads=4 > logs/ptest-$NAME-\$t.log 2>&1; echo \"tiny \$t rc \$? | \$(grep -aE '^test result' logs/ptest-$NAME-\$t.log)\"; grep -aE ' FAILED|panicked' logs/ptest-$NAME-\$t.log | head -3; done; \
  deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-cli --test docs_env_vars > logs/ptest-$NAME-docs_env_vars.log 2>&1; echo \"cli docs_env_vars rc \$? | \$(grep -aE '^test result' logs/ptest-$NAME-docs_env_vars.log)\"; \
  deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-core --lib > logs/ptest-$NAME-core.log 2>&1; echo \"core lib rc \$? | \$(grep -aE '^test result' logs/ptest-$NAME-core.log)\"; \
  docker rm -f ink-hip-specflip-build > /dev/null 2>&1" < /dev/null
