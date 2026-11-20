#!/usr/bin/env bash
# an.sh <ref ft> <cmp ft>...: analyze.py with the styles of the committed runs (run from anywhere)
R=/opt/code/intarsiaml/.worktrees/style7/docs/results/2026-11-20-style7/raw; S=/opt/code/intarsiaml/.worktrees/style7/docs/results/2026-11-19-ab35d-cutover-scoreboard/raw
ref=$1; shift; args=(); for c in "$@"; do args+=(--cmp $R/ft-$c.json); done
cd /opt/code/deepeval && uv run python /opt/code/intarsiaml/.worktrees/style7/scripts/style7/analyze.py --tok /opt/code/intarsiaml/.worktrees/style7/scratch/model/tokenizer.json --ref $R/ft-$ref.json "${args[@]}" --replies A1=$S/20261119-221213-adhoc-ab35d-A1-gsm8k/result.json B1=$S/20261119-221432-adhoc-ab35d-B1-gsm8k/result.json
