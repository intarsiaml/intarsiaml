#!/usr/bin/env bash
# Lane cmtp diagnosis ON spark1: which setting makes the served default's answers differ from --spec off's (userpath.sh FAIL)? The same two prompts
# (4k-0, short-0; max 128 tokens, temperature 0, natural stop) against servers that differ in --spec (off/default) and --profile-every (default/0).
# Answers are saved whole (out-diag/<tag>-<prompt>.txt) next to their sha256.
#   fleetq run --host spark1 --mem 50 --expect 25 --label "cmtp: diag" -- ssh spark1 'bash -s' < scripts/cmtp/diag.sh
set -uo pipefail
R=$HOME/intarsiaml-cmtp; H=/home/nvidia/.cache/huggingface/hub; O=$R/out-diag; mkdir -p $O
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/target-dev/release/intarsia}; PORT=8098; P=$R/src/scripts/cmtp/ids-prompts
ask() { python3 - "$1" $PORT "$2" <<'PY'
import json, sys, urllib.request, hashlib
p = open(sys.argv[1]).read()
body = json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 128, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{sys.argv[2]}/v1/chat/completions", body, {"content-type": "application/json"}), timeout=300)
d = json.load(r); c = d["choices"][0]["message"]["content"]
open(sys.argv[3], "w").write(c)
print(hashlib.sha256(c.encode()).hexdigest()[:16], d["usage"]["completion_tokens"], d["choices"][0]["finish_reason"])
PY
}
run() { local tag=$1; shift
  nohup $BIN serve --backend cuda --model $M35 --port $PORT --host 127.0.0.1 --max-seq 8192 --mem-budget 40 --no-supervisor "$@" > $O/serve-$tag.log 2>&1 < /dev/null &
  for i in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; sleep 3; done
  for f in 4k-0 short-0; do echo "$tag $f: $(ask $P/$f.txt $O/$tag-$f.txt)"; done
  pkill -f -- "--port $PORT"; sleep 8; pkill -9 -f -- "--port $PORT" 2>/dev/null; sleep 2
}
run off-p0 --spec off --profile-every 0
run def-p0 --profile-every 0
run off-pd --spec off
run def-pd
