#!/usr/bin/env bash
# Lane LDF, on the Mac, from the worktree root: copy the worktree to tatooine (a file copy, no fleet slot needed).
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .worktrees --exclude .mutbak --exclude '*-intarsia' --exclude ldf-out --exclude '/ldf-*' ./ tatooine:~/intarsiaml-ldf/
