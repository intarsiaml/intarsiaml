#!/bin/zsh
# Lane mchunkx: real shapes of every row-exact multi-row launch of a batched decode step. Kernel-mode profile (every launch its own command
# buffer, labelled kernel / t (rows) / n / k) of `serve --max-batch N` with S short streams, 24 tokens; then counts by (kernel, t, n, k).
# usage: shapes.sh <35b|27b> <max-batch> <streams> ; run under fleetq
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mchunkx; R=$W/docs/results/2026-11-17-mchunkx/raw/shapes; mkdir -p $R
BIN=${BIN:-$W/.base/bin-main/intarsia}
case $1 in 35b) M=$HOME/models/Qwen3.6-35B-A3B-NVFP4 ;; 27b) M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4 ;; esac
TXT=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
PORT=18691; O=$R/$1-mb$2-s$3; mkdir -p $O
INTARSIA_METAL_PROFILE=kernel:$O/prof.jsonl $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 8192 --profile-every 0 --no-supervisor --no-ui \
  --host 127.0.0.1 --port $PORT --max-batch $2 > $O/server.out 2> $O/server.err < /dev/null &
TP=$!
for i in {1..300}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $TP 2>/dev/null || break; sleep 1; done
curl -s http://127.0.0.1:$PORT/v1/intarsia/plan > $O/plan.json
python3 $W/scripts/mchunkx/served_ids.py run --url http://127.0.0.1:$PORT/v1 --text $TXT --out $O/ids.json --gen 24 --short --counts $3
kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
python3 - <<PY | tee $O/shapes.txt
import json,collections
c=collections.Counter()
for l in open("$O/prof.jsonl"):
    if l.startswith("{"):
        r=json.loads(l)
        if r["op"].endswith("_rows") or "gemv" in r["op"] or "moe_experts" in r["op"] or "logits" in r["op"]:
            c[(r["op"],r["t"],r["n"],r["k"])]+=1
for (op,t,n,k),v in sorted(c.items()): print(f"{op:36s} t={t:2d} n={n:7d} k={k:6d} count={v}")
PY
gzip -f $O/prof.jsonl $O/server.err
