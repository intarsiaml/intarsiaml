#!/usr/bin/env bash
# Lane mx: Metal no-op of the branch against origin/main on endor (after scripts/hg/metal-noop.sh): plan show + 256 greedy ids
# (Qwen3-8B, Qwen3.6-35B-A3B NVFP4; short and ~3.5k prompts) via scripts/hg/noop.sh, plan show of the 27B MXFP4 (its recipe
# qwen3_5.toml changed: HIP-only table), kernels test, workspace tests on the branch. Runs ON endor inside a fleetq job
# (--exclusive). Target dirs /opt/code/mx-targets/<t>-metal (APFS clones of hg's on first use).
set -uo pipefail
BASE=${BASE:-/opt/code/wt-mx-base}; DEV=${DEV:-/opt/code/wt-mx}
O=${OUT:-$DEV/docs/results/2026-10-25-mx-hip-27b-prefill/raw/metal}; mkdir -p "$O"
M8=$HOME/models/Qwen3-8B; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
mkdir -p /opt/code/mx-targets
for t in base dev; do
  [ -d /opt/code/mx-targets/$t-metal ] || cp -cR /opt/code/hg-targets/$t-metal /opt/code/mx-targets/$t-metal 2>/dev/null || true
  if [ $t = base ]; then T=$BASE; else T=$DEV; fi
  echo "== build $t $(git -C $T rev-parse --short HEAD) $(date +%T)"
  (cd $T && CARGO_TARGET_DIR=/opt/code/mx-targets/$t-metal cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  R=/opt/code/mx-targets/$t-metal/release/intarsia
  RUN="$R" bash $DEV/scripts/hg/noop.sh run $O $t metal $M8 $M35 $DEV/scripts/hg/prompts
  $R plan show --backend metal --model "$M27" --max-seq 8192 --json < /dev/null > "$O/plan-27-$t.json" 2> "$O/plan-27-$t.json.err"
  echo "plan 27 $t exit $?"
done
bash $DEV/scripts/hg/noop.sh cmp $O base dev
if cmp -s "$O/plan-27-base.json" "$O/plan-27-dev.json" && cmp -s "$O/plan-27-base.json.err" "$O/plan-27-dev.json.err"; then
  echo "plan 27 json: identical ($(wc -c < "$O/plan-27-dev.json" | tr -d ' ') B, err $(wc -c < "$O/plan-27-dev.json.err" | tr -d ' ') B)"
else echo "plan 27 json: DIFFER"; fi
echo "== kernels test metal (dev) $(date +%T)"
/opt/code/mx-targets/dev-metal/release/intarsia kernels test --backend metal < /dev/null 2>&1 | tail -3
[ -d /opt/code/mx-targets/test ] || cp -cR /opt/code/hg-targets/test /opt/code/mx-targets/test 2>/dev/null || true
echo "== workspace tests on the branch $(date +%T)"
(cd $DEV && CARGO_TARGET_DIR=/opt/code/mx-targets/test cargo test --workspace --release --no-fail-fast 2>&1 | grep -E "^test result|FAILED|failed|panicked" | sort | uniq -c | sort -rn | head -20)
echo "== done $(date +%T)"
