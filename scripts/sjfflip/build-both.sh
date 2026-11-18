#!/usr/bin/env bash
# sjfflip: build the origin/main merge base ("main") and HEAD ("new") CUDA binaries on spark1, main first (cold), new incrementally.
set -euo pipefail
D="$(dirname "${BASH_SOURCE[0]}")"
"$D/build.sh" main "$(git merge-base HEAD origin/main)"
"$D/build.sh" new HEAD
