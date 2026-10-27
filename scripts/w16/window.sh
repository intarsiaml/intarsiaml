#!/usr/bin/env bash
# Lane W16: does the Qwen3.8-27B MXFP4 run W4A16 with NO flags? Driven FROM THE MAC, one fleetq job that holds both R9700s and takes
# radiance down (the binary /work/w16-intarsia is built beforehand in ~/intarsiaml-w16 on tatooine, see docs/results/2026-10-27-w16-*):
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 25 --mem 30 \
#     --label "w16 gpu check" -- scripts/w16/window.sh
# Arms: a fresh `serve --tp 2` per arm (MTP k2 on by default at tp2): "default" (no --mxfp4-act), "none" (--mxfp4-act none), "dynamic"
# (--mxfp4-act dynamic). Each answers the six prompts of scripts/dfh/ids-prompts/short-*.txt greedily (temperature 0, 160 tokens);
# the answers' sha256 and the server's "mxfp4 activations:" line are the verdict: default == none (and its log says "recipe"), dynamic
# says "--mxfp4-act" and differs from none on at least one prompt. Radiance is restored by guard.sh when this script ends or dies.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-10-27-w16-w4a16-default/raw}; mkdir -p "$OUT"
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0; PORT=8097; SSH="ssh -o BatchMode=yes tatooine"
ARMS=${ARMS:-"default none dynamic"}
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
mkdir -p "$REPO/target"; FIFO=$(mktemp -u "$REPO/target/w16guard.XXXXXX"); mkfifo "$FIFO"
$SSH 'mkdir -p ~/w16' < /dev/null; scp -q "$HERE/guard.sh" tatooine:w16/
$SSH 'bash ~/w16/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
cleanup() { $SSH 'docker rm -f w16-serve' >/dev/null 2>&1 < /dev/null; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"; }
trap cleanup EXIT
serve() { # serve <arm>
  local flag=""
  case $1 in none) flag="--mxfp4-act none" ;; dynamic) flag="--mxfp4-act dynamic" ;; esac
  $SSH "docker rm -f w16-serve >/dev/null 2>&1; docker run -d --name w16-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0,1 \
    -v \$HOME/intarsiaml-w16:/work -v /data/models:/models:ro -w /work $IMG /work/w16-intarsia serve --backend hip --model $M27 \
    --served-model-name w16 --tp 2 --devices 0,1 --max-batch 1 --profile-every 0 --mem-budget 29 --max-seq 8192 $flag \
    --no-supervisor --host 0.0.0.0 --port $PORT" < /dev/null > "$OUT/docker-$1.id" 2>&1
  for i in $(seq 1 100); do curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
  $SSH 'docker logs --tail 60 w16-serve' > "$OUT/serve-fail-$1.log" 2>&1 < /dev/null; return 1
}
for arm in $ARMS; do
  say "arm $arm: starting server"
  if ! serve "$arm"; then say "arm $arm: server did not come up"; continue; fi
  for n in 0 1 2 3 4 5; do
    python3 - "$HERE/../dfh/ids-prompts/short-$n.txt" "$PORT" > "$OUT/ans-$arm-$n.json" 2>&1 <<'PY'
import json, sys, urllib.request
p = open(sys.argv[1]).read()
req = urllib.request.Request(f"http://tatooine:{sys.argv[2]}/v1/chat/completions",
    data=json.dumps({"model": "w16", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 160}).encode(),
    headers={"content-type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=180))
print(json.dumps({"content": d["choices"][0]["message"]["content"], "usage": d["usage"]}))
PY
  done
  $SSH 'docker logs w16-serve 2>&1 | grep -aE "mxfp4 activations|quantisation scheme note|speculative|mtp|error|tensor parallel"' < /dev/null > "$OUT/serve-$arm.log" 2>&1
  say "arm $arm: $(grep -a 'mxfp4 activations' "$OUT/serve-$arm.log" | head -1)"
  $SSH 'docker rm -f w16-serve' >/dev/null 2>&1 < /dev/null
done
python3 - "$OUT" $ARMS <<'PY'
import hashlib, json, sys
out, arms = sys.argv[1], sys.argv[2:]
h = {}
for a in arms:
    h[a] = []
    for n in range(6):
        try: h[a].append(hashlib.sha256(json.load(open(f"{out}/ans-{a}-{n}.json"))["content"].encode()).hexdigest()[:12])
        except Exception: h[a].append("ERR")
for a in arms: print(a, " ".join(h[a]))
if "default" in h and "none" in h: print("default == none:", h["default"] == h["none"] and "ERR" not in h["none"])
if "dynamic" in h and "none" in h: print("dynamic differs from none on", sum(x != y for x, y in zip(h["dynamic"], h["none"])), "of 6 prompts")
PY
