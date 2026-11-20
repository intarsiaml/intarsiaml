#!/usr/bin/env bash
# style7: live check that `serve`'s chat path renders the pinned prompts to vLLM's token count: 6 chat requests (thinking off), max_tokens=1, usage.prompt_tokens vs the /tokenize count. CPU backend on endor.
#   FLEETQ_OWNER=claude-style7 fleetq run --host endor --load cpu --mem 45 --expect 10 --label "style7 chat prompt_tokens check" -- bash scripts/style7/chatcount.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-style7/raw; BIN=$REPO/scratch/target/release/intarsia; M=$REPO/scratch/models/nvfp4; PORT=8195
$BIN serve --backend cpu --model $M --port $PORT --host 127.0.0.1 --served-model-name oracle --max-seq 2048 --mem-budget 40 --profile-every 0 --prefill-chunk 256 --spec off --no-supervisor > $OUT/serve-chatcount.log 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || { echo "serve died"; exit 1; }; sleep 2; done
python3 - "$OUT" "$PORT" <<'PY' | tee "$OUT/chat-prompt-tokens.txt"
import json, sys, urllib.request
out, port = sys.argv[1], sys.argv[2]
P = {r["id"]: r["prompt"] for r in json.load(open(out + "/gsm8k-pinned200-prompts.json"))}
T = {r["id"]: len(r["tokens"]) for r in json.load(open(out + "/vllm-tokenize-pinned200-think-off.json"))}
for i in ["gsm8k-00000", "gsm8k-00065", "gsm8k-00087", "gsm8k-00128", "gsm8k-00159", "gsm8k-00199"]:
    body = {"model": "oracle", "messages": [{"role": "user", "content": P[i]}], "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 1, "temperature": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    print(i, "ours prompt_tokens", r["usage"]["prompt_tokens"], "vllm /tokenize", T[i], "MATCH" if r["usage"]["prompt_tokens"] == T[i] else "DIFF")
PY
