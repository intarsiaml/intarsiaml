#!/bin/bash
# Lane dsv41/failpaths (docs/results/2026-11-03-dsv41-failpaths.md): the three failure paths of the DSv4.1 TP=2 pair on the Sparks.
# Run through fleetq over both Sparks as the printed commands say; `--dry-run` stubs every ssh and curl.
#
#   scripts/dsv41-tp2/failpaths-test.sh                print the build and test commands, run nothing
#   scripts/dsv41-tp2/failpaths-test.sh --dry-run      the cases' control flow with ssh and curl stubbed (local, no fleet)
#
# Run the printed commands in order. The test is one fleetq job over both Sparks: vLLM down for the window and always restored,
# memfine and the watcher with the 5 GiB kill line, all by run-full.sh, which sources this file as its HOOK.
#
# Cases (CASES, default "near force ring cancel supervised prefill golden kill"; "kill35" adds the 35B comparison). MAX_MIN bounds them, so the kills go last:
#   ring    chained cuts. A greedy reply of 14 tokens on a 200-token prompt (rows dropped behind it), the same prompt again, and a cold run of it:
#           the ids must match on the fix; `cached` shows the reuse refused in the band. On the baseline binary too: the ids may differ there.
#   cancel  a 12k-token prefill abandoned after 2 s by the client, then the same prompt: both ranks up, /health 200, ids equal to a cold run. Only this
#           shows the ranks in step after a cancel: the unit test has no collectives. `cached` above 0 on the retry shows the cancel landed mid-prefill.
#   supervised  a supervised `serve` (no --no-supervisor, default --watchdog-s 60), INTARSIA_DSV41_CAP=70000: a prefill of SUP_PROMPT (default 65000)
#           tokens and a temperature 0.7 reply of SUP_REPLY (default 3500) tokens, each longer than 60 s. Expected: HTTP 200, no `[supervisor]` restart,
#           /health 200. Main (BASE_SUPERVISED=1 to see it): exit 86. Memory at that capacity has not been measured: watch memfine.
#   near    #2. Greedy requests of cap-5, cap-4, cap-3, cap-2 tokens with max_tokens that just fit, one above the room (clamped), then a reply that fills
#           the capacity from below, at INTARSIA_DSV41_CAP=2048. Expected: HTTP 200, finish_reason length, both ranks up, /health 200. Main, by reading: both
#           ranks exit 87 at cap-4.
#   force   #3. FORCE_RUNS (default 3) times: rank 0 fails forward pass FORCE_PASS (default 30) of a temperature 0.7 reply with
#           INTARSIA_DSV41_TEST_RANK0_FAIL_PASS. Expected: HTTP 500, /health 503, both ranks exit 87.
#   prefill the cost of the device wait: base, fix, base, fix with fresh ranks, each timing PREFILL_RUNS (default 2) cold prompts of PREFILL_TOKENS (default
#           24000) tokens and one reply token at the same chunk (run-full.sh's INTARSIA_DSV41_CHUNK=512). In $OUT/prefill.tsv. Fix minus base includes the one message per chunk.
#   golden  the baseline binary (bin/intarsia.base, main) against the fix: chat texts (greedy, seeded temperature 0.7, a long prefill)
#           must be identical. The evidence that the normal path did not move.
#   kill    #1. KILLS (default 20) times: fresh ranks, a temperature 0.7 reply of 800 tokens, SIGKILL rank 0 at a random time KILL_MIN..KILL_MAX s
#           into it, at the default deadline (no INTARSIA_DSV41_RDMA_DEADLINE_S). Rank 1's exit code and seconds to exit, and which path took
#           it out, in $OUT/kill.tsv. Expected: 87 at about 7 s (drain 2 s + abort limit 3 s + detection), or 0 at 2 to 3 s when nothing was parked (2 of 20 on 2026-11-04).
#           BASE_KILLS (default 2) the same on the baseline binary: expected 34 s or more, not 87.
#   kill35  KILLS35 (default 4) kills of rank 0 of the Qwen3.6-35B at TP=2, temperature 0.7 (the 35B lane's numbers were greedy).
#
# Results: $OUT/steps.log, $OUT/kill.tsv, $OUT/kill35.tsv, $OUT/<case>/. Stops starting runs 4 min before MAX_MIN.

SELF="${BASH_SOURCE[0]}"
if [[ $SELF == "$0" && ${1:-} != --dry-run ]]; then
  R=$(cd "$(dirname "$0")/../.." && pwd); B=$(dirname "$R")/failpaths-base
  cat <<CMD
# 1. Binaries, vLLM stays up. The baseline first: both builds use the same SPARK_DIR, and the fix's build must be the last.
git -C $R worktree add $B 03f8dc2a3
SPARK_DIR=~/intarsia-dsv41-failpaths BIN_NAME=intarsia.base fleetq run --host spark2 --load cpu --mem 16 --expect 45 --label "dsv41 failpaths: baseline build" -- $B/scripts/dsv41-tp2/build.sh
SPARK_DIR=~/intarsia-dsv41-failpaths fleetq run --host spark2 --load cpu --mem 16 --expect 45 --label "dsv41 failpaths: fix build" -- $R/scripts/dsv41-tp2/build.sh
git -C $R worktree remove $B
# 2. The test. About 32 rank starts, guessed at 4.5 min each (load, kill, memory back), plus the supervised prefill and reply: MAX_MIN 210
#    bounds the cases, the vLLM restart comes on top. These times are guesses. Fewer kills (KILLS=10) save about 45 min.
SPARK_DIR=~/intarsia-dsv41-failpaths EARLYOOM_OK="3,2 2,1" STEPS="" MAX_MIN=210 HOOK=$R/scripts/dsv41-tp2/failpaths-test.sh OUT=$R/.scratch/failpaths-\$(date +%m%d-%H%M) \\
  fleetq run --host spark1,spark2 --quiet gpu,cpu --mem spark1=105,spark2=105 --takes-down spark1-vllm,spark2-vllm --expect 240 \\
  --label "dsv41 failpaths: 20 rank 0 kills, near-cap, forced failure, chained cuts, cancel, supervised 65k; vLLM restored after" -- $R/scripts/dsv41-tp2/run-full.sh
CMD
  exit 0
fi

# --dry-run: what run-full.sh provides, stubbed. kill -0 says a rank is gone and its exit file says 87, so every wait ends at once.
if [[ ${1:-} == --dry-run ]]; then
  DRY=$(cd "$(dirname "$0")/../.." && pwd)/.scratch/failpaths-dry; rm -rf "$DRY"; mkdir -p "$DRY"
  OUT=$DRY; SPARK_REL=x; SPARK_DIR='$HOME/x'; LOGD=l; WIP=w; WPORT=1; SPORT=2; SURL=http://stub; BIN=b; C0=c0; C1=c1; NCCLENV=""; DENV=""; TAG=t
  CASES=${CASES:-near force ring cancel supervised prefill golden kill kill35}; KILLS=2; SUP_PROMPT=10; SUP_REPLY=5; BASE_KILLS=1; KILLS35=1; FORCE_RUNS=1; KILL_MIN=0; KILL_MAX=1
  log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
  left() { echo 99999; }
  r() {
    echo "$1: $2" >> "$OUT/ssh.log"
    case $2 in
      *MemAvailable*) echo 100 ;;
      *.exit*) echo 87 ;;
      *pgrep*) echo 4242 ;;
      *'kill -0'*) return 1 ;;
    esac
    return 0
  }
  sleep() { :; }
  curl() { # the stubbed server: a JSON answer, HTTP 200
    local o="" w="" a
    for a in "$@"; do [[ $o == -w ]] && w=$a; o=$a; done
    [[ $* == */health* ]] && { echo -n 200; return 0; }
    echo '{"choices":[{"message":{"content":"stub"},"text":"stub","finish_reason":"length"}]}'
    [[ -n $w ]] && { echo; echo "HTTP 200"; }
    return 0
  }
fi

MY_CASES=${CASES:-near force ring cancel supervised prefill golden kill}
CASES=" " # failtest.sh: its helpers (chat1, finish, cleanup, text_of, ...) and none of its cases
source "$(dirname "$SELF")/failtest.sh"

BIN_FIX=$BIN; BIN_BASE="$SPARK_DIR/bin/intarsia.base"

# ranks under a wrapper that writes the rank's exit code (run-full.sh's start_ranks has none); the case's env as in failtest.sh
start_ranks() {
  TAG=$(date +%H%M%S)
  r spark2 "mkdir -p $LOGD; rm -f $LOGD/worker-$TAG.exit; ${R1PRE:-}nohup bash -c 'env $NCCLENV $DENV $1 ${R1ENV:-} $BIN tp-worker --listen $WIP:$WPORT -m $C1 --backend cuda --tp 2; echo \$? > $LOGD/worker-$TAG.exit' > $LOGD/worker-$TAG.log 2>&1 < /dev/null & echo \$! > $LOGD/worker-$TAG.pid"
  sleep 2
  r spark1 "mkdir -p $LOGD; rm -f $LOGD/serve-$TAG.exit; ${R0PRE:-}nohup bash -c 'env $NCCLENV $DENV $1 ${R0ENV:-} $BIN serve -m $C0 --backend cuda --tp 2 --tp-remote $WIP:$WPORT --port $SPORT --host 0.0.0.0 --served-model-name dsv41-intarsia ${SUPERVISE:---no-supervisor} --no-ui; echo \$? > $LOGD/serve-$TAG.exit' > $LOGD/serve-$TAG.log 2>&1 < /dev/null & echo \$! > $LOGD/serve-$TAG.pid"
  log "ranks started (tag $TAG, $BIN) env '$1' rank0 '${R0ENV:-}' rank1 '${R1ENV:-}'"
  wait_health 900 || exit 1
}
wait_health() { # <limit s>: up, or a rank left while loading
  local T0 up=0; T0=$(date +%s)
  while (( $(date +%s) - T0 < $1 )); do
    [[ $(curl -s -m 5 -o /dev/null -w '%{http_code}' "$SURL/health") == 200 ]] && { up=1; break; }
    [[ -n $(exitcode spark1 serve) ]] && { log "rank 0 exited while loading"; break; }
    [[ -n $(exitcode spark2 worker) ]] && { log "rank 1 exited while loading"; break; }
    sleep 5
  done
  log "serve up=$up after $(( $(date +%s) - T0 )) s"
  [[ $up == 1 ]]
}
# the ranks' own PIDs: the wrappers' children
pids() {
  P0=$(r spark1 "pgrep -P \$(cat $LOGD/serve-$TAG.pid) | head -1")
  P1=$(r spark2 "pgrep -P \$(cat $LOGD/worker-$TAG.pid) | head -1")
  log "pids: rank 0 (spark1) ${P0:-none}, rank 1 (spark2) ${P1:-none}"
}
exitcode() { # <host> <serve|worker> -> the exit code, or nothing while the rank runs
  r "$1" "cat $LOGD/$2-$TAG.exit 2>/dev/null; true"
}
sampled_json() { # <max_tokens> <temperature> <seed> <prompt>
  python3 -c '
import json, sys
print(json.dumps({"model": "dsv41-intarsia", "messages": [{"role": "user", "content": sys.argv[4]}], "temperature": float(sys.argv[2]), "seed": int(sys.argv[3]), "max_tokens": int(sys.argv[1])}))' "$@"
}
sampled_bg() { # <name> <max_tokens> <temperature> <seed> <prompt>
  sampled_json "$2" "$3" "$4" "$5" > "$OUT/$1.req"
  ( s=$(now_ms); curl -s -m 300 -w '\nHTTP %{http_code}\n' "$SURL/v1/chat/completions" -H 'content-type: application/json' -d @"$OUT/$1.req" > "$OUT/$1.resp" 2>"$OUT/$1.curlerr"; rc=$?; echo "$s $(now_ms) $rc" > "$OUT/$1.times" ) &
  REQ_PID=$!
}
http_of() { grep -a '^HTTP' "$1" 2>/dev/null | tail -1 | cut -d' ' -f2; }
random_s() { python3 -c "import random, sys; print('%.2f' % random.uniform(float(sys.argv[1]), float(sys.argv[2])))" "$1" "$2"; }
# wait for a rank's exit after the fault at $TF: seconds, then the code, on two lines; "<limit>" and "none" if it is still up after <limit> s
wait_exit() { # <host> <serve|worker> <limit s>
  local end=$(( $(date +%s) + $3 )) c
  while (( $(date +%s) < end )); do
    c=$(exitcode "$1" "$2")
    [[ -n $c ]] && { python3 -c "import sys; print('%.1f' % ((int(sys.argv[1]) - int(sys.argv[2])) / 1000))" "$(now_ms)" "$TF"; echo "$c"; return; }
    sleep 0.3
  done
  echo "$3"; echo none
}

case_golden() {
  local O=$OUT0/golden arm f v s
  CASE=golden
  for arm in fix base; do
    BIN=$BIN_FIX; [[ $arm == base ]] && BIN=$BIN_BASE
    OUT=$O/$arm; mkdir -p "$OUT"
    start_ranks ""; pids
    chat1 g-cap 48 "$P_CAP"; chat1 g-cnt 64 "$P_CNT"; chat1 g-code 200 "$P_CODE"; chat1 g-long 400 "$P_LONG"; chat1 g-pref 16 "@$OUT0/prefill-prompt.txt"
    ids_req i-200 200 100 256; ids_req i-12k 12000 100 64 # token ids, not only text; 12000 rows cross the prefill's device waits (one per 4096 rows)
    for s in 1 2; do
      sampled_json 200 0.7 $s "$P_LONG" > "$OUT/s$s.req"
      curl -s -m 300 "$SURL/v1/chat/completions" -H 'content-type: application/json' -d @"$OUT/s$s.req" > "$OUT/s$s.resp"
      text_of "$OUT/s$s.resp" > "$OUT/s$s.txt"
    done
    finish "$arm"
  done
  for f in g-cap g-cnt g-code g-long g-pref s1 s2; do
    cmp -s "$O/fix/$f.txt" "$O/base/$f.txt" && v=identical || v=DIFFERENT
    log "RESULT golden $f: fix vs baseline $v"
  done
  for f in i-200 i-12k; do
    log "RESULT golden ids $f: fix vs baseline $(ids_same "$O/fix/$f.ids" "$O/base/$f.ids") ($(wc -w < "$O/fix/$f.ids") words)"
  done
  BIN=$BIN_FIX; OUT=$OUT0
}

case_kill() { # <binary> <runs> <tsv>
  local bin=$1 n=$2 tsv=$3 i d res code path
  CASE=kill; OUT=$OUT0/kill; mkdir -p "$OUT"
  [[ -s $tsv ]] || printf 'binary\trun\tkill_after_s\trank1_exit\trank1_s\tpath\tcurl_rc\n' > "$tsv"
  for i in $(seq 1 "$n"); do
    [[ $(left) -lt 240 ]] && { log "out of time before kill run $i"; break; }
    BIN=$bin; start_ranks ""; pids
    sampled_bg "$(basename "$bin")-$i" 800 0.7 "$i" "$P_LONG"
    d=$(random_s "${KILL_MIN:-2}" "${KILL_MAX:-15}")
    sleep "$d"
    TF=$(now_ms); r spark1 "kill -KILL $P0"
    res=$(wait_exit spark2 worker "${KILL_LIMIT:-200}")
    code=${res##*$'\n'}; res=${res%%$'\n'*}
    wait "$REQ_PID"
    r spark2 "tail -n 60 $LOGD/worker-$TAG.log" > "$OUT/$(basename "$bin")-$i.worker.log" 2>&1
    path=$(grep -aoE 'disconnected during a pass|queued work did not finish|queued work failed|serving stopped' "$OUT/$(basename "$bin")-$i.worker.log" | head -1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(basename "$bin")" "$i" "$d" "$code" "$res" "${path:-none}" "$(cut -d' ' -f3 "$OUT/$(basename "$bin")-$i.times" 2>/dev/null)" >> "$tsv"
    log "RESULT kill $(basename "$bin") $i: killed at $d s, rank 1 exit $code after $res s (${path:-no path logged})"
    cleanup
  done
  BIN=$BIN_FIX; OUT=$OUT0
}

near_req() { # <name> <prompt tokens> <max_tokens>: /v1/completions on token ids, temperature 0, no EOS stop
  python3 -c '
import json, sys
print(json.dumps({"model": "dsv41-intarsia", "prompt": [100 + i % 1000 for i in range(int(sys.argv[1]))], "temperature": 0, "max_tokens": int(sys.argv[2]), "ignore_eos": True}))' "$2" "$3" > "$OUT/$1.req"
  curl -s -m 300 -w '\nHTTP %{http_code}\n' "$SURL/v1/completions" -H 'content-type: application/json' -d @"$OUT/$1.req" > "$OUT/$1.resp" 2>"$OUT/$1.curlerr"
  local h; h=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "$SURL/health")
  log "RESULT near $1 (prompt $2, max_tokens $3): HTTP $(http_of "$OUT/$1.resp"), /health $h, rank 0 exit '$(exitcode spark1 serve)', rank 1 exit '$(exitcode spark2 worker)', $(python3 -c '
import json, sys
try:
    d = json.loads(open(sys.argv[1]).read().split("\nHTTP")[0]); print("finish", d["choices"][0]["finish_reason"])
except Exception as e:
    print("no json")' "$OUT/$1.resp")"
}
case_near() {
  local cap=2048
  CASE=near; OUT=$OUT0/near; mkdir -p "$OUT"
  start_ranks "INTARSIA_DSV41_CAP=$cap"; pids
  near_req below $(( cap - 5 )) 5
  near_req at4 $(( cap - 4 )) 4 # max_tokens that fit: valid on main too, so the case shows draft_fits and not the clamp
  near_req at3 $(( cap - 3 )) 3
  near_req at2 $(( cap - 2 )) 2
  near_req clamp $(( cap - 4 )) 64
  near_req fill $(( cap - 300 )) 300
  near_req well 1000 64
  finish near
  OUT=$OUT0
}

case_force() {
  local i h res0 res1
  CASE=force; OUT=$OUT0/force; mkdir -p "$OUT"
  for i in $(seq 1 "${FORCE_RUNS:-3}"); do
    local R0ENV="INTARSIA_DSV41_TEST_RANK0_FAIL_PASS=${FORCE_PASS:-30}" # local: the later cases must not inherit the fault
    start_ranks ""; pids
    TF=$(now_ms)
    sampled_bg "f$i" 400 0.7 "$i" "$P_LONG"
    wait "$REQ_PID"
    h=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "$SURL/health")
    res0=$(wait_exit spark1 serve 60); res1=$(wait_exit spark2 worker 60)
    log "RESULT force $i: HTTP $(http_of "$OUT/f$i.resp") ($(head -c 160 "$OUT/f$i.resp" | tr '\n' ' ')), /health $h; rank 0: $(echo $res0), rank 1: $(echo $res1) (seconds after the request started, exit code); expected 500, 503, 87, 87"
    finish "f$i"
  done
  OUT=$OUT0
}

ids_req() { # <name> <prompt tokens> <first id> <max_tokens> [curl limit s]: /v1/completions on token ids, temperature 0, no EOS stop; ids in $OUT/<name>.ids
  python3 -c '
import json, sys
print(json.dumps({"model": "dsv41-intarsia", "prompt": [int(sys.argv[2]) + i % 1000 for i in range(int(sys.argv[1]))], "temperature": 0, "max_tokens": int(sys.argv[3]), "ignore_eos": True, "return_token_ids": True}))' "$2" "$3" "$4" > "$OUT/$1.req"
  curl -s -m "${5:-600}" -w '\nHTTP %{http_code}\n' "$SURL/v1/completions" -H 'content-type: application/json' -d @"$OUT/$1.req" > "$OUT/$1.resp" 2>"$OUT/$1.curlerr"
  python3 -c '
import json, sys
try:
    d = json.loads(open(sys.argv[1]).read().split("\nHTTP")[0]); c = d["choices"][0]
    print(" ".join(str(i) for i in c.get("token_ids", [])), "| cached", (d.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens"))
except Exception as e:
    print("<no ids: %s>" % type(e).__name__)' "$OUT/$1.resp" > "$OUT/$1.ids"
}
ids_same() { [[ "$(sed 's/ | cached.*//' "$1")" == "$(sed 's/ | cached.*//' "$2")" ]] && echo identical || echo DIFFERENT; }

case_ring() {
  local arm
  CASE=ring
  for arm in fix base; do
    OUT=$OUT0/ring/$arm; mkdir -p "$OUT"
    BIN=$BIN_FIX; [[ $arm == base ]] && BIN=$BIN_BASE
    start_ranks ""; pids
    ids_req a 200 100 14     # rows dropped behind the reply
    ids_req b 200 100 64     # the same prompt: 15 tokens cut, 18 or more below the furthest row when 3 or more rows were dropped
    ids_req u 600 2000 4     # an unrelated prompt: the sequence restarts
    ids_req cold 200 100 64
    log "RESULT ring $arm: reply after the chained cut vs cold $(ids_same "$OUT/b.ids" "$OUT/cold.ids") (b $(sed 's/.*| //' "$OUT/b.ids"), cold $(sed 's/.*| //' "$OUT/cold.ids"))"
    finish "$arm"
  done
  BIN=$BIN_FIX; OUT=$OUT0
}

case_cancel() {
  CASE=cancel; OUT=$OUT0/cancel; mkdir -p "$OUT"
  start_ranks ""; pids
  ids_req gone 12000 100 16 2 # the client leaves during the prefill
  ids_req retry 12000 100 16
  ids_req u 600 2000 4
  ids_req cold 12000 100 16
  log "RESULT cancel: retry vs cold $(ids_same "$OUT/retry.ids" "$OUT/cold.ids") (retry $(sed 's/.*| //' "$OUT/retry.ids")), /health $(curl -s -m 2 -o /dev/null -w '%{http_code}' "$SURL/health"), rank 0 exit '$(exitcode spark1 serve)', rank 1 exit '$(exitcode spark2 worker)'"
  finish cancel
  OUT=$OUT0
}

case_supervised() { # <binary> <name>: rank 0's PID here is the supervisor's; the child leaves with it (its parent check)
  CASE=supervised; OUT=$OUT0/supervised-$2; mkdir -p "$OUT"
  BIN=$1; SUPERVISE=" "
  start_ranks "INTARSIA_DSV41_CAP=70000"; pids
  local t0; t0=$(date +%s)
  ids_req prefill "${SUP_PROMPT:-65000}" 100 8 900
  local took=$(( $(date +%s) - t0 )) stalled v
  stalled=$(r spark1 "grep -ac 'exit 86\|restart #' $LOGD/serve-$TAG.log; true")
  # a prefill under the watchdog shows nothing either way: only one that outlasts it would be killed with ticks that follow the enqueue
  v=PASS; (( took <= 60 )) && v="INCONCLUSIVE (under the 60 s watchdog: raise SUP_PROMPT)"; [[ ${stalled:-0} != 0 || $(http_of "$OUT/prefill.resp") != 200 ]] && v=FAIL
  log "RESULT supervised $2 prefill of ${SUP_PROMPT:-65000} tokens: $v, HTTP $(http_of "$OUT/prefill.resp") after $took s, /health $(curl -s -m 2 -o /dev/null -w '%{http_code}' "$SURL/health"), watchdog or restart lines $stalled"
  t0=$(date +%s)
  sampled_json "${SUP_REPLY:-3500}" 0.7 1 "$P_LONG" > "$OUT/reply.req"
  curl -s -m 900 -w '\nHTTP %{http_code}\n' "$SURL/v1/chat/completions" -H 'content-type: application/json' -d @"$OUT/reply.req" > "$OUT/reply.resp" 2>"$OUT/reply.curlerr"
  log "RESULT supervised $2 reply of ${SUP_REPLY:-3500} tokens: HTTP $(http_of "$OUT/reply.resp") after $(( $(date +%s) - t0 )) s, /health $(curl -s -m 2 -o /dev/null -w '%{http_code}' "$SURL/health"), supervisor restarts: $(r spark1 "grep -ac 'restart #' $LOGD/serve-$TAG.log; true")"
  finish supervised
  SUPERVISE=""; BIN=$BIN_FIX; OUT=$OUT0
}

case_prefill() {
  local tsv=$OUT0/prefill.tsv arm i k t name
  CASE=prefill; OUT=$OUT0/prefill; mkdir -p "$OUT"
  printf 'order\tarm\trun\tseconds\thttp\tprefill_chunk\n' > "$tsv"
  k=0
  for arm in base fix base fix; do
    k=$(( k + 1 )); BIN=$BIN_FIX; [[ $arm == base ]] && BIN=$BIN_BASE
    start_ranks ""; pids
    for i in $(seq 1 "${PREFILL_RUNS:-2}"); do
      name=p$k-$i
      python3 -c '
import json, sys
print(json.dumps({"model": "dsv41-intarsia", "prompt": [int(sys.argv[2]) + i % 1000 for i in range(int(sys.argv[1]))], "temperature": 0, "max_tokens": 1, "ignore_eos": True, "return_token_ids": True}))' "${PREFILL_TOKENS:-24000}" $(( 3000 + i * 1000 )) > "$OUT/$name.req"
      t=$(curl -s -m 600 -o "$OUT/$name.resp" -w '%{time_total} %{http_code}' "$SURL/v1/completions" -H 'content-type: application/json' -d @"$OUT/$name.req")
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$k" "$arm" "$i" "${t%% *}" "${t##* }" "$(python3 -c '
import json, sys
try:
    print(json.load(open(sys.argv[1]))["usage"]["prompt_tokens_details"].get("prefill_chunk"))
except Exception:
    print("?")' "$OUT/$name.resp")" >> "$tsv"
      log "RESULT prefill $arm (order $k) run $i: ${t%% *} s, HTTP ${t##* }"
    done
    finish "prefill-$k"
  done
  log "RESULT prefill mean s: $(awk -F'\t' 'NR>1{s[$2]+=$4; n[$2]++} END{for(a in s) printf "%s %.2f  ", a, s[a]/n[a]}' "$tsv")"
  BIN=$BIN_FIX; OUT=$OUT0
}

case_near_base() { # the defect itself, on the baseline binary: expected both ranks exit 87 at cap-4
  local cap=2048 res0 res1
  CASE=near-base; OUT=$OUT0/near-base; mkdir -p "$OUT"; BIN=$BIN_BASE
  start_ranks "INTARSIA_DSV41_CAP=$cap"; pids
  TF=$(now_ms)
  near_req at4 $(( cap - 4 )) 4
  res0=$(wait_exit spark1 serve 60); res1=$(wait_exit spark2 worker 60)
  log "RESULT near baseline at cap-4: rank 0 $(echo $res0), rank 1 $(echo $res1) (seconds, exit code)"
  finish near-base
  BIN=$BIN_FIX; OUT=$OUT0
}

case_kill35() { # the 35B on the fix's binary: same kill, its own flags (11 GB a rank)
  local M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
  local FL="-m $M35 --backend cuda --tp 2 --max-seq 8192 --mem-budget 20" tsv=$OUT0/kill35.tsv i d res code
  CASE=kill35; OUT=$OUT0/kill35; mkdir -p "$OUT"
  printf 'run\tkill_after_s\trank1_exit\trank1_s\n' > "$tsv"
  for i in $(seq 1 "${KILLS35:-4}"); do
    [[ $(left) -lt 240 ]] && { log "out of time before kill35 run $i"; break; }
    TAG=q$(date +%H%M%S)
    r spark2 "rm -f $LOGD/worker-$TAG.exit; nohup bash -c 'env $NCCLENV $BIN_FIX tp-worker --serving --listen $WIP:$WPORT $FL; echo \$? > $LOGD/worker-$TAG.exit' > $LOGD/worker-$TAG.log 2>&1 < /dev/null & echo \$! > $LOGD/worker-$TAG.pid"
    sleep 2
    r spark1 "rm -f $LOGD/serve-$TAG.exit; nohup bash -c 'env $NCCLENV $BIN_FIX serve $FL --tp-remote $WIP:$WPORT --port $SPORT --host 0.0.0.0 --served-model-name dsv41-intarsia --no-supervisor --no-ui; echo \$? > $LOGD/serve-$TAG.exit' > $LOGD/serve-$TAG.log 2>&1 < /dev/null & echo \$! > $LOGD/serve-$TAG.pid"
    wait_health 420 || exit 1
    pids
    sampled_bg "q$i" 800 0.7 "$i" "$P_LONG"
    d=$(random_s "${KILL_MIN:-2}" "${KILL_MAX:-8}")
    sleep "$d"
    TF=$(now_ms); r spark1 "kill -KILL $P0"
    res=$(wait_exit spark2 worker 60); code=${res##*$'\n'}; res=${res%%$'\n'*}
    wait "$REQ_PID"
    printf '%s\t%s\t%s\t%s\n' "$i" "$d" "$code" "$res" >> "$tsv"
    log "RESULT kill35 $i: killed at $d s, rank 1 exit $code after $res s"
    cleanup
  done
  OUT=$OUT0
}

for c in $MY_CASES; do
  [[ $(left) -lt 240 ]] && { log "out of time before case $c"; break; }
  case $c in
    near) case_near; case_near_base ;;
    prefill) case_prefill ;;
    force) case_force ;;
    ring) case_ring ;;
    cancel) case_cancel ;;
    supervised) case_supervised "$BIN_FIX" fix; [[ ${BASE_SUPERVISED:-0} == 1 ]] && case_supervised "$BIN_BASE" base ;;
    golden) case_golden ;;
    kill) case_kill "$BIN_FIX" "${KILLS:-20}" "$OUT0/kill.tsv"; case_kill "$BIN_BASE" "${BASE_KILLS:-2}" "$OUT0/kill.tsv" ;;
    kill35) case_kill35 ;;
    *) log "unknown case $c" ;;
  esac
done
if [[ -n ${DRY:-} ]]; then # the forced fault belongs to case force alone
  L=$(grep -rl FAIL_PASS --include=ssh.log "$DRY" | grep -v '/force/'); [[ -z $L ]] || { log "dry run: the fault leaked into $L"; exit 1; }
fi
log "failpaths: done; results in $OUT0"
