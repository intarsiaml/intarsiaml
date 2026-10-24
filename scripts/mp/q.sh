#!/usr/bin/env bash
# Lane mp: queue a dev.sh run on a Spark and print one compact line per variant.
#   scripts/mp/q.sh <host> <expect-min> "<label>" <T list> <reps> <skew> [variant ...]     env ONLY=v3|prod passes through
H=$1; EXP=$2; LBL=$3; shift 3
export FLEETQ_OWNER=${FLEETQ_OWNER:-claude-f31ccf91-mp}
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${MPOUT:-/tmp/mp_q.$$.out}
fleetq run --host "$H" --quiet gpu,cpu --expect "$EXP" --label "mp: $LBL" -- "$D/dev.sh" "$H" "$@" > "$OUT" 2>&1
python3 - "$OUT" <<'PY'
import sys,re
cur=None; row={}
def flush():
    if cur: print("%-18s %s" % (cur, "  ".join("%s:%s" % (k,v) for k,v in row.items())))
for l in open(sys.argv[1]):
    m=re.match(r"== (\S+) flags=",l)
    if m:
        flush(); cur=m.group(1); row={}; continue
    p=l.split()
    if len(p)>=4 and p[0].isdigit() and p[1] in("prod","v3"):
        tag=("" if p[1]=="v3" else "P")
        bw="" if p[-1] in("ref",) else ""
        row["%s%s"%(tag,p[0])]=p[2]+(("!" if "DIFF" in l else "") if p[1]=="v3" and "prod" in open(sys.argv[1]).read().split("== ",2)[1] else "")
    elif "error" in l or "status" in l: print(l.rstrip())
flush()
PY
