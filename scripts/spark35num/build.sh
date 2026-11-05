#!/usr/bin/env bash
# Lane spark35: sync a tree to $HOST ~/intarsiaml-s35/src and build the CUDA CLI (+ test binaries), inside a fleetq slot.
#   build.sh <ref|WORKTREE> <binary name> [test <crate> <test> ...]
#   ref = git ref (git archive of that commit, e.g. origin/main) or WORKTREE (this checkout, committed state is not required).
# One cargo target dir; the second build is incremental.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ref=$1; name=$2; shift 2
rsh "mkdir -p $L/src $L/bin $L/out"
# a binary already built from this commit is kept (the sources' mtimes are refreshed below, which would force a full rebuild)
if [[ $ref != WORKTREE ]]; then
  SHA=$(git -C "$REPO" rev-parse "$ref")
  if rsh "test \"\$(cat $L/bin/$name.ref 2>/dev/null)\" = $SHA && test -x $L/bin/$name"; then echo "$name is already built from $SHA"; exit 0; fi
fi
if [[ $ref == WORKTREE ]]; then
  STAMP="wt-$("$REPO/scripts/tree-stamp.sh")"; HEADSHA=$(git -C "$REPO" rev-parse --short HEAD); echo "$HEADSHA dirty=$(git -C "$REPO" status --porcelain | wc -l)" > "$RES/head-$name.txt"
  rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .venv --exclude .claude --exclude .worktrees --exclude .scratch --exclude out "$REPO/" "$HOST:intarsiaml-s35/src/"
else
  SHA=$(git -C "$REPO" rev-parse "$ref"); STAMP="ref-${SHA:0:12}"
  rsh "rm -rf $L/stage && mkdir -p $L/stage"
  git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$HOST" "tar -x -C $L/stage"
  rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $L/src/ && rm -rf $L/stage"
fi
rsh "cd $L/src && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
echo "build $name from $ref ($STAMP) $(date +%T)"
rsh "rm -f $L/src/target/release/intarsia; cd $L/src && env CARGO_BUILD_JOBS=${JOBS:-16} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|warning: unused|Finished|could not' | tail -6; cp $L/src/target/release/intarsia $L/bin/$name && $L/bin/$name --version | head -1"
rsh "test -x $L/bin/$name" || { echo "BUILD FAILED: $name"; exit 1; }
[[ ${SHA:-} ]] && rsh "echo $SHA > $L/bin/$name.ref"
echo "built $name $(date +%T)"
rsh "df -h ~ | tail -1; du -sh $L | tail -1"
