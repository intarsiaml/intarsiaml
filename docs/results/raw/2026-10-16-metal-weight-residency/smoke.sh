#!/bin/zsh
# One slot: (1) residency-set refresh period sweep, (2) keepwarm tests, (3) serve idle A/B base | kwb | rs, round 1 checked, 6 rounds.
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/resid
R=/opt/code/intarsiaml-residency/docs/results/raw/2026-10-16-metal-weight-residency
$R/probe2.sh $R/probe5 1 "residencycpu 40000 1000 x" "residencycpu 40000 2000 x" "residencycpu 40000 5000 x" "residencycpu 10000 100000 x" "residencycpu 3000 100000 x" "residencycpu 10000 200 x"
$SC/keepwarm-test --test-threads=1 > $R/keepwarm-test.txt 2>&1; echo "keepwarm test exit $?"
R0=1 $R/serve-idle.sh $R/idle1 1 base kwb rs
python3 - $R/idle1 <<'PY' || exit 1
import json,glob,sys
fs=glob.glob(sys.argv[1]+"/*.idle*.json"); ok=len(fs)==9
for f in fs:
    try: j=json.loads(open(f).read().strip().splitlines()[-1]); ok&= j["status"]==200 and j["ttft_ms"] is not None
    except Exception: ok=False
print("round-1 check", ok, len(fs)); sys.exit(0 if ok else 1)
PY
cp -r $R/idle1 $R/idle6 2>/dev/null
R0=2 $R/serve-idle.sh $R/idle6 6 base kwb rs
