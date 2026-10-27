#!/usr/bin/env bash
# Lane W4A8, ON tatooine inside a window that holds both cards (radiance down; cat-window.sh runs it as PRE=pre4):
#   1. FUSE: the fused-quant decode variant (INTARSIA_HIP_MXFP4_A8_FUSE=1) on card 0: reference + M-invariance test, the output hash
#      with the knob off and on (must be equal), decode timings with the knob on (rows_exact timing_mxfp4_rows).
#   2. ids: the MTP ids gate of the 27B at tp2 with --mxfp4-act fp8, through a server: --spec off against the shipped default (MTP k2)
#      on the 18 prompts of scripts/cmtp/ids-prompts, 256 forced tokens, temperature 0; every answer must be identical.
# Logs: ~/intarsiaml-w4a8-logs/pre4 (copied back by cat-window.sh).
set -uo pipefail
R=$HOME/intarsiaml-w4a8; V=intarsia-cargo-w4a8; D=$R/deploy/tatooine/dev.sh
O=${O:-$HOME/intarsiaml-w4a8-logs/pre4}; mkdir -p "$O"
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
PORT=8096
now() { date +%T; }
dev() { local g=$1; shift; INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V HIP_VISIBLE_DEVICES=$g $D "$@" < /dev/null; }

if [[ ${SKIP_FUSE:-0} != 1 ]]; then
  echo "$(now) FUSE"
  F='^test |^test result|FAILED|panicked|FUSEHASH|^TIME'
  for fz in 0 1; do
    echo "-- mxfp4_a8_fuse FUSE=$fz"
    dev 0 env INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_MXFP4_A8_FUSE=$fz cargo test --release -p intarsia-backend-hip --test mxfp4_a8_fuse -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F"
  done
  echo "-- decode timing, FUSE=0 then 1 (rows_exact timing_mxfp4_rows, W4A8 kernels only)"
  for fz in 0 1; do
    echo "FUSE=$fz"
    dev 0 env INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_MXFP4_A8_FUSE=$fz cargo test --release -p intarsia-backend-hip --test rows_exact timing_mxfp4_rows -- --ignored --nocapture 2>&1 | grep -aE "^TIME" | grep -a "_a8_"
  done
fi

if [[ ${SKIP_IDS:-0} != 1 ]]; then
  serve() { # serve <tag> [extra serve args]
    local tag=$1; shift
    docker rm -f w4a8-ids >/dev/null 2>&1
    docker run -d --name w4a8-ids --network host --device /dev/kfd --device /dev/dri \
      --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video --ipc host \
      --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0,1 \
      -v "$R":/work -v /data/models:/models:ro -w /work $IMG /work/w4a8-intarsia serve --backend hip --model $M27 \
      --served-model-name ids-$tag --tp 2 --devices 0,1 --mxfp4-act fp8 --max-batch 1 --profile-every 0 --mem-budget 29 --max-seq 20480 \
      --no-supervisor --host 0.0.0.0 --port $PORT "$@" > "$O/docker-$tag.id" 2>&1 < /dev/null
    for i in $(seq 1 200); do curl -sf -m 3 "http://localhost:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
    docker logs --tail 40 w4a8-ids > "$O/serve-fail-$tag.log" 2>&1; return 1
  }
  ask() { # ask <tag> : the 18 prompts -> $O/ans-<tag>.tsv (name, sha256 of the text, completion tokens)
    python3 - "$1" "$PORT" "$R/scripts/cmtp/ids-prompts" "$O/ans-$1.tsv" <<'PY'
import json, sys, hashlib, urllib.request, os
tag, port, pdir, out = sys.argv[1:5]
names = [f"short-{i}" for i in range(6)] + [f"4k-{i}" for i in range(6)] + [f"16k-{i}" for i in range(6)]
rows = []
for n in names:
    prompt = open(os.path.join(pdir, n + ".txt")).read()
    body = json.dumps({"model": "ids-" + tag, "prompt": prompt, "max_tokens": 256, "min_tokens": 256, "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/completions", body, {"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        text = d["choices"][0]["text"]
        rows.append((n, hashlib.sha256(text.encode()).hexdigest()[:16], d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"]))
    except Exception as e:
        rows.append((n, "ERROR " + str(e)[:60], 0, 0))
open(out, "w").write("".join("\t".join(map(str, r)) + "\n" for r in rows))
PY
  }
  for arm in off auto; do
    echo "$(now) ids arm $arm: starting server"
    if [[ $arm == off ]]; then serve off --spec off; else serve auto; fi || { echo "serve $arm failed"; tail -5 "$O/serve-fail-$arm.log"; continue; }
    ask $arm
    curl -s -m 5 "http://localhost:$PORT/metrics" | grep -aE "spec_(proposed|accepted)" | head -4 > "$O/metrics-$arm.txt"
    docker rm -f w4a8-ids >/dev/null 2>&1
    echo "$(now) ids arm $arm done"
  done
  echo "-- ids verdict"
  python3 - "$O" <<'PY'
import sys
o = sys.argv[1]
def load(t):
    try:
        return {l.split("\t")[0]: l.rstrip("\n").split("\t")[1:] for l in open(f"{o}/ans-{t}.tsv")}
    except FileNotFoundError:
        return {}
a, b = load("off"), load("auto")
ok = 0
for n in a:
    same = n in b and a[n][0] == b[n][0] and not a[n][0].startswith("ERROR") and a[n][1] == "256"
    ok += same
    print(("PASS" if same else "FAIL"), n, a[n], b.get(n))
print(f"ids {ok}/{len(a)} identical (off vs MTP default)")
PY
  cat "$O/metrics-auto.txt" 2>/dev/null
fi
