#!/bin/sh
# ink-pfgap2 step 1: GPU idle between prefill chunks on current main (cb trace = GPUStartTime / GPUEndTime per command buffer).
# One serve process (--spec off, default chunk), prompts of ~4k and ~16k tokens, every request unique (no prefix reuse),
# max_tokens 1. Per request: back-to-back (0 s idle before) and after 5 s idle. Request times on the trace clock (CLOCK_UPTIME_RAW).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
W=/opt/code/intarsiaml/.worktrees/ink-pfgap2; O=${GAP_OUT:-$W/.scratch/gap}; mkdir -p $O
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
B=${GAP_BIN:-$W/target/release/intarsia}; [ -x $B ] || { echo "GUARD: no binary"; exit 1; }
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
C=$W/crates/intarsia-models/examples/data/bench_corpus.txt
PORT=8197
echo "tree $(git -C $W rev-parse --short=9 HEAD) $(git -C $W status --porcelain --untracked-files=no | wc -l | tr -d ' ') changed bin $B $(date +%T)"
SP=
trap '[ -n "$SP" ] && kill $SP 2>/dev/null; wait 2>/dev/null' EXIT TERM INT
now() { python3 -c 'import time; print(time.clock_gettime(time.CLOCK_UPTIME_RAW))'; }
# mk <chars> <offset> <tag> -> $O/req.json
mk() { python3 - "$C" "$1" "$2" "$3" $O/req.json <<'PY'
import json, sys
t = open(sys.argv[1]).read(); n = int(sys.argv[2]); off = int(sys.argv[3])
json.dump({"model": "inkling-2-mini", "prompt": "Request " + sys.argv[4] + ".\n" + t[off:off + n] + " Summarize the text above.", "max_tokens": 1, "temperature": 0}, open(sys.argv[5], "w"))
PY
}
post() { curl -s -o $O/last.out -w "%{time_total}" http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' -d @$O/req.json; }
env INTARSIA_METAL_PROFILE=cb:$O/cb.jsonl $B serve --model $CK --backend metal --spec off --host 127.0.0.1 --port $PORT --max-seq 20480 --served-model-name inkling-2-mini > $O/serve.log 2>&1 &
SP=$!
up=0; for i in $(seq 1 400); do curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && { up=1; echo "server up after $i s"; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "GUARD: server not up"; tail -8 $O/serve.log; exit 1; }
sleep 3
: > $O/requests.tsv
# warm-up: one uncached 4k and one 16k (first request of a process pays one-time costs)
mk 12500 0 warm4; post > /dev/null; sleep 2
mk 46500 0 warm16; post > /dev/null; sleep 2
n=0
# size:chars:idle-before-seconds; offsets differ per request, prefixes differ too
for spec in 4k:12500:0 4k:12500:0 4k:12500:0 4k:12500:5 4k:12500:5 4k:12500:5 16k:46500:0 16k:46500:0 16k:46500:5 16k:46500:5; do
  sz=${spec%%:*}; r=${spec#*:}; ch=${r%%:*}; idle=${r#*:}
  n=$((n + 1)); mk $ch $((n * 700 % 90000)) "r$n"
  sleep $idle
  s=$(now); t=$(post); e=$(now)
  tok=$(python3 -c "import json;u=json.load(open('$O/last.out'))['usage'];print(u['prompt_tokens'],u['prompt_tokens_details']['cached_tokens'])")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $n $sz $idle $s $e $t "$tok" | tee -a $O/requests.tsv
done
kill $SP; wait $SP 2>/dev/null; SP=
echo "== done $(date +%T)"
