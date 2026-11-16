#!/bin/bash
# Lane mhkv: with MTP off, the fix changes nothing a served Metal server shows or stores. Per bin (main, fix; pinned under
# target/mhkv-run/), a fresh `serve --spec off --prefix-store ...` on an empty store dir: GET /v1/intarsia/plan, the 4k P
# request, wait for the disk writer, stop. Then: the plans byte-identical; the store's object payloads (each .bin file past
# its 48-byte header, which holds the build key) the same multiset; every manifest without a `head` field. Also the plan of
# a server with the MTP default (`--spec` unset; the fix adds no op, so it must be identical too).
# usage (inside a fleetq slot): scripts/mhkv/mtpoff.sh 35b|27b
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mhkv; RUN=${MHKV_RUN:-$W/target/mhkv-run}; PR=$RUN/prompts; RAW=${MHKV_RAW:-$W/docs/results/2026-11-16-mhkv/raw}
case $1 in
  35b) M=$HOME/models/Qwen3.6-35B-A3B-NVFP4 ;;
  27b) M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4 ;;
  *) echo "model?"; exit 2 ;;
esac
[ -f $PR/P-4k.json ] || python3 $W/scripts/mhkv/prompts.py $PR > /dev/null
R=$RAW/mtpoff-$1; mkdir -p $R
S=$RUN/store-mtpoff-$1; rm -rf $S; mkdir -p $S
PORT=18642; URL=http://127.0.0.1:$PORT
start() { # bin name -- args
  local b=$1 n=$2; shift 2
  $RUN/intarsia-$b serve --backend metal --model $M --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT "$@" > $R/$n.server.out 2> $R/$n.server.err < /dev/null &
  SP=$!
  for i in {1..300}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo "SERVER DIED $n"; tail -5 $R/$n.server.err; return 1; }; sleep 1; done
}
stop() {
  kill -INT $SP; for i in {1..60}; do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
}
for b in main fix; do
  start $b off-$b --spec off --prefix-store "host_mb=8192,disk_mb=65536,dir=$S/$b" || exit 1
  curl -sf $URL/v1/intarsia/plan > $R/plan-off-$b.json
  python3 $W/scripts/mhkv/req.py $URL $PR/P-4k.json off-$b-P 64 | tee -a $R/requests.jsonl
  for i in {1..240}; do
    p=$(curl -sf $URL/v1/intarsia/prefix-store | python3 -c 'import json,sys; d=json.load(sys.stdin); s=d.get("stats",d); print(s.get("disk_pending_bytes",0), (s.get("disk") or {}).get("bytes",0))' 2>/dev/null)
    [ "${p%% *}" = 0 ] && [ "${p##* }" != 0 ] && break; sleep 0.25
  done
  curl -sf $URL/v1/intarsia/prefix-store > $R/store-off-$b.json
  stop
  start $b on-$b || exit 1
  curl -sf $URL/v1/intarsia/plan > $R/plan-on-$b.json
  stop
done
for k in off on; do
  if cmp -s $R/plan-$k-main.json $R/plan-$k-fix.json; then echo "plan MTP-$k: byte-identical ($(wc -c < $R/plan-$k-main.json | tr -d ' ') bytes)"; else echo "plan MTP-$k: DIFFERS"; diff $R/plan-$k-main.json $R/plan-$k-fix.json | head -10; fi
done
python3 - $S <<'EOF'
import hashlib, json, os, sys
root = sys.argv[1]
def scan(b):
    objs, mans = [], []
    for dp, _, fs in os.walk(os.path.join(root, b)):
        for f in fs:
            d = open(os.path.join(dp, f), "rb").read()
            if f.endswith(".bin"):
                objs.append((len(d), hashlib.sha256(d[48:]).hexdigest()))
            elif f.endswith(".entry"):
                mans.append(json.loads(d[24:]))
    return sorted(objs), mans
(om, mm), (of, mf) = scan("main"), scan("fix")
print(f"store MTP-off: main {len(om)} objects / {len(mm)} entries, fix {len(of)} / {len(mf)}; payloads identical: {om == of}; "
      f"manifest fields main {sorted({k for m in mm for k in m})} fix {sorted({k for m in mf for k in m})}; "
      f"token lists identical: {sorted(m['tokens'] for m in mm) == sorted(m['tokens'] for m in mf)}")
EOF
