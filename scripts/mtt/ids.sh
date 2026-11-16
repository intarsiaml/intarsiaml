#!/bin/zsh
# Lane mtt gate 1: served greedy ids, lazy admission on (default) == off (INTARSIA_METAL_MTP_TTFT_FIX=0) == `--spec off`; the MTP
# counters and the server's `spec:` line identical between on and off. usage (fleetq slot): OUT=dir ids.sh MODEL_DIR TAG
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${W:-/opt/code/intarsiaml/.worktrees/mtt}; BIN=${BIN:-$W/target/metal/release/intarsia}; R=${OUT:?}; mkdir -p $R; PORT=${PORT:-18613}
M=$1; T=$2; P=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for arm in on old off; do
  EV=(); FL=(); [ $arm = old ] && EV=(INTARSIA_METAL_MTP_TTFT_FIX=0); [ $arm = off ] && FL=(--spec off)
  env X=1 $EV $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT $FL > $R/$T-$arm.out 2> $R/$T-$arm.err < /dev/null &
  TP=$!; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  echo "## $T $arm: $(grep -a '^loaded' $R/$T-$arm.err | sed 's/.* in //;s/,.*//'), $(grep -a '^admission' $R/$T-$arm.err)"
  python3 $W/scripts/mtt/ids.py http://127.0.0.1:$PORT/v1 $R/$T-$arm.json $P/prompt-4096.txt $P/prompt-16384.txt
  kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
  grep -a "spec:" $R/$T-$arm.err | tail -1
done
python3 - $R $T <<'PY'
import json, sys
r, t = sys.argv[1:3]; d = {a: json.load(open(f"{r}/{t}-{a}.json")) for a in ("on", "old", "off")}
ok = True
for c in d["on"]:
    same = d["on"][c]["ids"] == d["old"][c]["ids"] == d["off"][c]["ids"]; spec = d["on"][c]["spec"] == d["old"][c]["spec"]
    ok &= same and spec and len(d["on"][c]["ids"]) == 256
    print(f"GATE {t} {c}: ids on==old==off {same} ({len(d['on'][c]['ids'])} tokens), MTP counters on==old {spec} {d['on'][c]['spec']}")
print(f"GATE {t}: {'PASS' if ok else 'FAIL'}")
PY
