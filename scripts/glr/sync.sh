#!/usr/bin/env bash
# Lane GLR, on the Mac, from the worktree root: copy the worktree to tatooine (a file copy, no fleet slot needed).
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .worktrees --exclude .mutbak --exclude '*-intarsia' --exclude glr-out --exclude '/glr-*' ./ tatooine:~/intarsiaml-glr/
