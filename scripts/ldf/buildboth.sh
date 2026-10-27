#!/usr/bin/env bash
# Lane LDF, on the Mac from the worktree root: build the release CLI of origin/main (base) and of this tree (dev) on tatooine, each as a fleetq
# cpu job, plus the dev mxfp4_a8 test binary. Copies land in ~/intarsiaml-ldf/{ldf-base-cli,ldf-dev-cli} and ldf-bin/mxfp4_a8-dev.
set -uo pipefail
EX=(--exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .worktrees --exclude .mutbak --exclude '*-intarsia' --exclude ldf-out --exclude '/ldf-*' --exclude '*-cli' --exclude bench-out)
git fetch -q origin
rm -rf ldf-out/basetree; mkdir -p ldf-out/basetree; git archive origin/main | tar -x -C ldf-out/basetree
rsync -a --delete "${EX[@]}" ldf-out/basetree/ tatooine:~/intarsiaml-ldf/ || exit 1
rsync -a scripts/ldf/ tatooine:~/intarsiaml-ldf/scripts/ldf/ || exit 1
fleetq run --host tatooine-r9700 --load cpu --expect 20 --label "GLR build base cli (origin/main $(git rev-parse --short origin/main))" -- ssh -o BatchMode=yes tatooine 'bash ~/intarsiaml-ldf/scripts/ldf/build.sh ldf-base-cli' 2>&1 | tail -6
rsync -a --delete "${EX[@]}" ./ tatooine:~/intarsiaml-ldf/ || exit 1
fleetq run --host tatooine-r9700 --load cpu --expect 20 --label "GLR build dev cli + test binary" -- ssh -o BatchMode=yes tatooine 'bash ~/intarsiaml-ldf/scripts/ldf/build.sh ldf-dev-cli && bash ~/intarsiaml-ldf/scripts/ldf/tbuild.sh dev' 2>&1 | tail -6
