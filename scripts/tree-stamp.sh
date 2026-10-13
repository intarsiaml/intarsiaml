#!/usr/bin/env bash
# Identify the code a measurement ran on: HEAD commit, a hash of the working tree's code (docs excluded, so editing a
# results doc does not change it), and whether the working tree differs from HEAD. Uses a throwaway index, so the
# real index and the commit history are untouched. Run it in the worktree right before syncing to the build box.
#   scripts/tree-stamp.sh        ->  head=a7da13b code-tree=1f2e3d4c5b6a dirty=yes
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
CODE=(crates kernels deploy tools scripts recipes src web Cargo.toml Cargo.lock rustfmt.toml pyproject.toml)
PATHS=(); for p in "${CODE[@]}"; do [[ -e "$p" ]] && PATHS+=("$p"); done
IDX="$(mktemp)"; trap 'rm -f "$IDX"' EXIT
export GIT_INDEX_FILE="$IDX"
git read-tree HEAD
git add -A -- "${PATHS[@]}" >/dev/null 2>&1
tree="$(git write-tree)"
unset GIT_INDEX_FILE
headtree="$(git rev-parse "HEAD^{tree}")"
# code-tree of HEAD for the same paths, to decide "dirty"
IDX2="$(mktemp)"; trap 'rm -f "$IDX" "$IDX2"' EXIT
GIT_INDEX_FILE="$IDX2" git read-tree HEAD
head_code="$(GIT_INDEX_FILE="$IDX2" git write-tree)"
dirty=no; [[ "$tree" != "$head_code" ]] && dirty=yes
echo "head=$(git rev-parse --short=7 HEAD) code-tree=${tree:0:12} dirty=$dirty"
