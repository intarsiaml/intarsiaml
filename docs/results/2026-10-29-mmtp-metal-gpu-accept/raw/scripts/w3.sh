#!/bin/bash
# MMTP window 3: gates. Inside a fleetq slot.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mmtp; O=$W/.lane/${OUT:-w3}; mkdir -p $O
NEW=$W/target/release/intarsia; MAIN=$W/.lane/main-target/release/intarsia
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; PORT=18474
cd $W
echo "## plan show $(date +%T)"
for b in NEW MAIN; do ${!b} plan show --model $M --backend metal --spec mtp > $O/plan-$b.txt 2>&1; echo "plan $b exit $?"; done
diff $O/plan-MAIN.txt $O/plan-NEW.txt > $O/plan.diff; echo "plan diff lines $(wc -l < $O/plan.diff)"
echo "## kernels test $(date +%T)"
$NEW kernels test --backend metal > $O/kernels-test-metal.log 2>&1; echo "kernels test --backend metal exit $?"
$NEW kernels test --backend metal --model $M > $O/kernels-test-metal-35b.log 2>&1; echo "kernels test --backend metal --model 35B exit $?"
# served: prefix reuse vs cold, batched vs solo, all greedy MTP
req() { # out prompt-file suffix
  python3 - "$1" "$2" "$3" $PORT <<'PY'
import json, sys, urllib.request
out, pf, suf, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
body = {"model": "intarsia", "prompt": open(pf).read() + suf, "max_tokens": 256, "temperature": 0, "ignore_eos": True}
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
j = json.load(r); open(out, "w").write(j["choices"][0]["text"]); print(out, j.get("usage"))
PY
}
up() { env X=1 "$@" > $O/srv-$tag.out 2> $O/srv-$tag.err & sp=$!; for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done; }
down() { kill -INT $sp; wait $sp 2>/dev/null; for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done; }
S="serve --backend metal --model $M --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT --spec mtp"
P=$W/.lane/prompts
echo "## served warm (A, A again, B) $(date +%T)"
tag=warm; up $NEW $S
req $O/warm-A1.txt $P/4ka.txt ""; req $O/warm-A2.txt $P/4ka.txt ""; req $O/warm-B.txt $P/4ka.txt " Summarize the above in one paragraph."
down
echo "## served cold B $(date +%T)"
tag=cold; up $NEW $S
req $O/cold-B.txt $P/4ka.txt " Summarize the above in one paragraph."
down
echo "## served batched (--max-batch 8, A and B together) $(date +%T)"
tag=batch; up $NEW $S --max-batch 8
req $O/batch-A.txt $P/4ka.txt "" & a=$!
req $O/batch-B.txt $P/4ka.txt " Summarize the above in one paragraph." & b=$!
wait $a $b
down
grep -c "verify passes\|spec" $O/srv-batch.err
for x in "warm-A1 warm-A2" "warm-B cold-B" "warm-A1 batch-A" "cold-B batch-B"; do set -- $x; cmp -s $O/$1.txt $O/$2.txt && echo "$1 == $2" || echo "$1 != $2 DIFFERS"; done
echo "## served done $(date +%T)"
echo "## cargo gates $(date +%T)"
INTARSIA_TEST_REAL_MODEL=1 cargo test --release -p intarsia-models --test batch_metal_real -- --test-threads 1 > $O/batch_metal_real.log 2>&1; echo "batch_metal_real exit $?"; grep -E "^test |test result" $O/batch_metal_real.log
cargo test --release -p intarsia-models --test batch_recurrent > $O/batch_recurrent.log 2>&1; echo "batch_recurrent exit $?"; grep -E "test result" $O/batch_recurrent.log
echo "## gates done $(date +%T)"
