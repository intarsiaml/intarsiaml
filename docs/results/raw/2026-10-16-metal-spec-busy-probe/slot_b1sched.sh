#!/bin/zsh
# B=1 schedule check: the speculation trace (`spec:` lines, INTARSIA_SPEC_TRACE=1) of ONE request at a time must be byte-identical
# between the base binary (5cd03a5) and the new one, both for `--max-batch 1` and `--max-batch 8` (a lone request through the
# batching scheduler). Fixed prompts (no nonce), greedy, 512 tokens; token ids are compared too.
#   usage: slot_b1sched.sh OUTDIR      (inside a fleetq slot)    env: BIN_NEW, BIN_BASE (required), PORT
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=${1:?outdir}; mkdir -p $O; O=${O:A}
BIN_NEW=${BIN_NEW:?}; BIN_BASE=${BIN_BASE:?}; PORT=${PORT:-8143}
W=/opt/code/intarsiaml-specprobe; M=$HOME/models/Qwen3-8B; URL=http://127.0.0.1:$PORT
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
SPID=""
stop_server() { if [ -n "$SPID" ] && kill -0 $SPID 2>/dev/null; then kill $SPID 2>/dev/null; for i in {1..20}; do kill -0 $SPID 2>/dev/null || break; sleep 0.5; done; kill -9 $SPID 2>/dev/null; fi; SPID=""; }
trap stop_server EXIT
trap "exit 143" INT TERM
echo "slot $FLEETQ_JOB $(date); new $BIN_NEW $(shasum -a 256 $BIN_NEW | cut -c1-16); base $BIN_BASE $(shasum -a 256 $BIN_BASE | cut -c1-16)"
for mb in 1 8; do
  for who in base new; do
    if [ $who = base ]; then BIN=$BIN_BASE; else BIN=$BIN_NEW; fi
    INTARSIA_SPEC_TRACE=1 $BIN serve --model $M --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT --max-batch $mb --spec ngram > $O/server-$who-mb$mb.log 2>&1 &
    SPID=$!
    ok=0; for i in {1..180}; do curl -sf $URL/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 1; done
    [ $ok = 1 ] || { echo "server $who mb$mb failed"; tail -5 $O/server-$who-mb$mb.log; stop_server; continue; }
    python3 - $URL $W/tools/prompts $O/ids-$who-mb$mb.json <<'PY'
import json, sys, urllib.request
url, prompts, out = sys.argv[1:4]
res = {}
for kind, f in (("grounded", "m3_grounded.txt"), ("open", "m3_open.txt"), ("grounded2", "m3_grounded.txt"), ("open2", "m3_open.txt")):
    text = open(f"{prompts}/{f}").read().strip()
    body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": 512, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "return_token_ids": True, "ignore_eos": kind.startswith("open")}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    res[kind] = {"ids": r["choices"][0].get("token_ids"), "usage": r.get("usage")}
json.dump(res, open(out, "w"))
print({k: (len(v["ids"] or []), v["usage"]) for k, v in res.items()})
PY
    curl -s $URL/metrics > $O/metrics-$who-mb$mb.txt
    stop_server
  done
  grep -a "^spec: " $O/server-base-mb$mb.log > $O/spec-base-mb$mb.txt; grep -a "^spec: " $O/server-new-mb$mb.log > $O/spec-new-mb$mb.txt
  echo "mb=$mb: spec lines base $(wc -l < $O/spec-base-mb$mb.txt) new $(wc -l < $O/spec-new-mb$mb.txt)"
  cmp $O/spec-base-mb$mb.txt $O/spec-new-mb$mb.txt && echo "mb=$mb: spec trace IDENTICAL"
  cmp $O/ids-base-mb$mb.json $O/ids-new-mb$mb.json && echo "mb=$mb: token ids IDENTICAL"
done
pgrep -fl "intarsia.* serve" || echo "no intarsia serve left"
echo "done $(date)"
