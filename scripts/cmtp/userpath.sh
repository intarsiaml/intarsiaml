#!/usr/bin/env bash
# Lane cmtp user path ON spark1: the 35B served the way a user starts it (supervised, no --spec, no --max-batch, default --profile-every), with and
# without --prefix-store on, against a --spec off server without the store: the same three requests (a ~3k-token prompt twice, so the second one is
# a store hit, then a short one) must answer byte-identically (a --spec off server with the store is reported too); the default servers must have run speculation (intarsia_spec_steps_total > 0) and
# print the default line, the off server must not.
#   fleetq run --host spark1 --mem 50 --expect 25 --label "cmtp: user path" -- ssh spark1 'bash -s' < scripts/cmtp/userpath.sh
set -uo pipefail
R=$HOME/intarsiaml-cmtp; H=/home/nvidia/.cache/huggingface/hub; O=$R/out-user; mkdir -p $O
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/target-dev/release/intarsia}; PORT=8097; P=$R/src/scripts/cmtp/ids-prompts
ask() { # ask <promptfile> -> sha256 of the answer text
  python3 - "$1" $PORT <<'PY'
import json, sys, urllib.request, hashlib
p = open(sys.argv[1]).read()
body = json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 128, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{sys.argv[2]}/v1/chat/completions", body, {"content-type": "application/json"}), timeout=300)
d = json.load(r); c = d["choices"][0]["message"]["content"]
print(hashlib.sha256(c.encode()).hexdigest()[:16], d["usage"]["completion_tokens"], repr(c[:40]))
PY
}
run() { # run <tag> <args...>
  local tag=$1; shift
  nohup $BIN serve --backend cuda --model $M35 --port $PORT --host 127.0.0.1 --max-seq 8192 --mem-budget 40 "$@" > $O/serve-$tag.log 2>&1 < /dev/null &
  for i in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; sleep 3; done
  for f in 4k-0 4k-0 short-0; do echo "$tag $f: $(ask $P/$f.txt)"; done | tee $O/answers-$tag.txt
  echo "$tag spec steps: $(curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_spec_steps_total' | sed 's/{[^}]*}//' | tr '\n' ' ')"
  echo "$tag default line: $(grep -ac 'default of recipe qwen3_5_moe on cuda at tp=1' $O/serve-$tag.log)"
  pkill -f -- "--port $PORT"; sleep 8; pkill -9 -f -- "--port $PORT" 2>/dev/null; sleep 2
}
run def-plain
run def-store --prefix-store on
run off-plain --spec off
run off-store --prefix-store on --spec off
a=$(cut -d' ' -f3 $O/answers-def-plain.txt | tr '\n' ' '); b=$(cut -d' ' -f3 $O/answers-def-store.txt | tr '\n' ' '); c=$(cut -d' ' -f3 $O/answers-off-plain.txt | tr '\n' ' '); d=$(cut -d' ' -f3 $O/answers-off-store.txt | tr '\n' ' ')
[ "$a" = "$c" ] && [ "$b" = "$c" ] && echo "PASS user path: default (plain, store) answers == spec-off without a store ($c)" || echo "FAIL user path: $a | $b | $c"
[ "$d" = "$c" ] && echo "INFO spec-off with the store == spec-off without" || echo "INFO spec-off WITH the prefix store differs from spec-off without it ($d vs $c): the store changes prefill numerics on its own, with or without MTP"
