#!/usr/bin/env bash
# Lane GLR, on the Mac from the worktree root: build the release CLI of origin/main (base) and of this tree (dev) on tatooine, each as a fleetq
# cpu job, plus the dev mxfp4_a8 test binary. Copies land in ~/intarsiaml-glr/{glr-base-cli,glr-dev-cli} and glr-bin/mxfp4_a8-dev.
set -uo pipefail
EX=(--exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .worktrees --exclude .mutbak --exclude '*-intarsia' --exclude glr-out --exclude '/glr-*' --exclude '*-cli' --exclude bench-out)
git fetch -q origin
rm -rf glr-out/basetree; mkdir -p glr-out/basetree; git archive origin/main | tar -x -C glr-out/basetree
rsync -a --delete "${EX[@]}" glr-out/basetree/ tatooine:~/intarsiaml-glr/ || exit 1
rsync -a scripts/glr/ tatooine:~/intarsiaml-glr/scripts/glr/ || exit 1
fleetq run --host tatooine-r9700 --load cpu --expect 20 --label "GLR build base cli (origin/main $(git rev-parse --short origin/main))" -- ssh -o BatchMode=yes tatooine 'bash ~/intarsiaml-glr/scripts/glr/build.sh glr-base-cli' 2>&1 | tail -6
rsync -a --delete "${EX[@]}" ./ tatooine:~/intarsiaml-glr/ || exit 1
fleetq run --host tatooine-r9700 --load cpu --expect 20 --label "GLR build dev cli + test binary" -- ssh -o BatchMode=yes tatooine 'bash ~/intarsiaml-glr/scripts/glr/build.sh glr-dev-cli && bash ~/intarsiaml-glr/scripts/glr/tbuild.sh dev' 2>&1 | tail -6
