#!/bin/zsh
# Lane mstore: `plan show --json` of the pinned binary (no crate change) against the branch binary, 8B / 27B / 35B, knobs unset.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mstore; R=$W/docs/results/2026-11-15-mstore-metal/raw/plan-show; mkdir -p $R
A=$W/target/mstore-run/intarsia-measure; B=$W/target/fix-target/release/intarsia
for m in Qwen3-8B Qwen3.8-27B-Quark-AWQ-MXFP4 Qwen3.6-35B-A3B-NVFP4; do
  $A plan show --model $HOME/models/$m --backend metal --json > $R/$m.base.json 2> $R/$m.base.err
  $B plan show --model $HOME/models/$m --backend metal --json > $R/$m.branch.json 2> $R/$m.branch.err
  if cmp -s $R/$m.base.json $R/$m.branch.json; then echo "$m: byte-identical ($(wc -c < $R/$m.base.json) bytes)"; else echo "$m: DIFFERS"; diff $R/$m.base.json $R/$m.branch.json | head -10; fi
done
