#!/usr/bin/env bash
# Lane W4A8, on the Mac, from the worktree root: copy the worktree to tatooine (no fleet slot needed; a file copy).
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .worktrees --exclude .mutbak ./ tatooine:~/intarsiaml-w4a8/
