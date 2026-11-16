#!/usr/bin/env bash
# Lane q27-plan, the radiance-down window (STEPS selects; default all):
#   kern   `kernels test --backend hip --quick`
#   plan   `plan show --admit --json` on the 27B (the plan stage's command, tp=1, the recipe's default spec), then the text table
#   qual   `qualify` on the 27B with every stage skipped but doctor, recipe, plan, kernels (the 27B's quick golden given only
#          because the CLI wants one; no accuracy stage runs)
set -u
. "$(dirname "$0")/lib.sh"
cd "$TREE"
STEPS=${STEPS:-"kern plan qual"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
W=/work/${O#"$TREE"/}
M27=/models/Qwen3.8-27B-Quark-AWQ-MXFP4
BN=qwen3.8-27b-mxfp4-hfquark
log "stages start; $STAMP; STEPS=$STEPS"
if has kern; then run 0 900 kernels $BIN kernels test --backend hip --quick; tail -3 "$O/kernels.out" | tee -a "$O/steps.log"; fi
if has plan; then
  run 0 600 plan-json $BIN plan show --model $M27 --backend hip --mem-budget 29 --admit --json
  run 0 600 plan-text $BIN plan show --model $M27 --backend hip --mem-budget 29 --admit
  python3 - "$O/plan-json.out" <<'PY' | tee -a "$O/steps.log"
import json, sys
d = json.load(open(sys.argv[1]))
bad = [o for o in d["ops"] if o["admitted"] is not True]
print("plan ops:", len(d["ops"]), "unadmitted:", [(o["op"], o["kernel"]) for o in bad])
for o in d["ops"]:
    if o["kernel"].endswith("_rows"):
        print("  rows kernel:", o["op"], o["kernel"], "admitted", o["admitted"])
PY
fi
if has qual; then
  SK="--skip gate --skip acc-format --skip acc-short --skip acc-oracle --skip bench --skip acc-long --skip acc-4k --skip envelope --skip bench-long --skip ab --skip tp"
  run 0 1200 qual $BIN qualify --backend hip --model $M27 --model-name Qwen3.8-27B-Quark-AWQ-MXFP4 --golden /work/golden/$BN-top20.safetensors \
    --mem-budget 29 --out $W/qual $SK
  grep -aE "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" "$O/qual.err" "$O/qual.out" | cut -c1-260 | tee -a "$O/steps.log"
fi
log "stages done"
