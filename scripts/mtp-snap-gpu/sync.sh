#!/usr/bin/env bash
# Lane mtp-snap-gpu (ON THE MAC): copy the PR worktree (`sync.sh pr`) or origin/main (`sync.sh main`) to tatooine:~/intarsiaml-mtpsnap-gpu/ and write its .stamp.
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:?pr|main}" in
  pr)
    scripts/tree-stamp.sh > .stamp
    rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .lane --exclude docs/results ./ tatooine:intarsiaml-mtpsnap-gpu/
    ;;
  main)
    ref=${MAIN_REF:-origin/main}
    ssh tatooine 'mkdir -p ~/intarsiaml-mtpsnap-gpu && cd ~/intarsiaml-mtpsnap-gpu && find . -mindepth 1 -maxdepth 1 -exec rm -rf {} +'
    git archive --format=tar "$ref" -- Cargo.toml Cargo.lock rustfmt.toml rust-toolchain.toml pyproject.toml crates kernels deploy tools scripts recipes src web golden 2>/dev/null \
      | ssh tatooine 'cd ~/intarsiaml-mtpsnap-gpu && tar -xm'   # -m: mtime = now (git archive stamps the commit time, and cargo would trust the old build artifacts)
    rsync -a scripts/mtp-snap-gpu/ tatooine:intarsiaml-mtpsnap-gpu/scripts/mtp-snap-gpu/   # the lane's own scripts are not on main
    echo "head=$(git rev-parse --short=7 "$ref") ref=$ref code-tree=$(git rev-parse "$ref^{tree}" | cut -c1-12) dirty=no" | ssh tatooine 'cat > ~/intarsiaml-mtpsnap-gpu/.stamp'
    ;;
esac
ssh tatooine 'cat ~/intarsiaml-mtpsnap-gpu/.stamp'
