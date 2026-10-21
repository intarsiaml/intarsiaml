#!/bin/zsh
# G3 A/B x3, interleaved (base dev | dev base | base dev), fresh `serve` per arm, `intarsia bench speed --at-most`,
# sizes 128 / 1k / 4k / 16k, gen 128, 3 reps, macmon 250 ms per arm. base = origin/main 4be7b7d4, dev = perf/metal-g3.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
TAG=$1
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad
OUT=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687/docs/results/raw/2026-10-21-g3-metal/$TAG; mkdir -p $OUT
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
SPID=""; MPID=""
trap '[[ -n "$SPID" ]] && kill $SPID 2>/dev/null; [[ -n "$MPID" ]] && kill $MPID 2>/dev/null' EXIT
echo "base $(md5 -q $S/intarsia-base) dev $(md5 -q $S/intarsia-dev) $(date -Iseconds)" | tee $OUT/info.txt
run() { # label bin
  local label=$1 bin=$2
  echo "##### $label $(date -Iseconds)"
  $bin serve --model $M --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 20000 \
    --profile-every 0 --no-supervisor --no-ui > $OUT/$label-server.out 2> $OUT/$label-server.err &
  SPID=$!
  for i in {1..240}; do curl -sf http://127.0.0.1:8093/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || break; sleep 1; done
  /Users/josef/bin/macmon pipe -i 250 > $OUT/$label-macmon.jsonl 2>/dev/null & MPID=$!
  $S/intarsia-base bench speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096,16384 --gen 128 --reps 3 \
    --thinking off --at-most --gb-per-token 15.1368 --bandwidth-gbs 614 --prefill-chunk 2048 --label $label --out-dir $OUT > $OUT/$label-bench.txt 2>&1
  echo "bench exit $?"; cat $OUT/$label-bench.txt | tail -8
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null; MPID=""
  kill $SPID 2>/dev/null; for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && kill -9 $SPID; wait $SPID 2>/dev/null; SPID=""
  for i in {1..30}; do lsof -iTCP:8093 -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done
  sleep 5
}
run base-1 $S/intarsia-base; run dev-1 $S/intarsia-dev
run dev-2 $S/intarsia-dev;   run base-2 $S/intarsia-base
run base-3 $S/intarsia-base; run dev-3 $S/intarsia-dev
echo AB_DONE $(date -Iseconds)
