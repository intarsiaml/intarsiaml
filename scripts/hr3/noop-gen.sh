#!/usr/bin/env bash
# Lane hr3: 256 greedy ids of the 35B after a ~1.2k-token prompt, for the no-op check (inside the HIP dev container, repo root = /work).
# Usage: scripts/hr3/noop-gen.sh [extra generate flags, e.g. --spec mtp --spec-k 2]. Prints the ids line and the spec statistics.
set -uo pipefail
M=/models/Qwen3.6-35B-A3B-NVFP4
err=$(mktemp)
/target/release/intarsia generate -m $M --backend hip --max-seq 4096 --chat --no-think --temperature 0 --profile-every 0 --ignore-eos \
  --show-ids -p "$(head -c 5000 tools/prompts/corpus_docs.txt)" --max-tokens 256 "$@" < /dev/null > /dev/null 2> $err
grep -a '^ids:' $err
grep -a -E '^spec:|--spec' $err | cut -c1-200
rm -f $err
