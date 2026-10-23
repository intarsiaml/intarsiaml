#!/usr/bin/env bash
# gc2: Metal 35B prefill chunk 256 -> 2048 (recipes/qwen3_5_moe.toml [tuning.metal]). ONE fleetq job on endor:
#   fleetq run --host endor --mem 42 --expect 60 --quiet gpu,cpu --label "gc2: ..." -- run.sh OUTDIR
# Order (each step is skipped when it no longer fits before DEADLINE): 8B/35B plan show main vs branch, qualify q-0 (old 35B
# Metal baselines moved aside; records new ones at chunk 2048), install them, qualify q-1 (verdict), 256 greedy ids main vs
# branch, prefill tok/s at 4k (chunk 256 = main binary, 2048 = branch binary).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
OUT=${1:?outdir}; mkdir -p "$OUT"; O=$(cd "$OUT" && pwd)
cd "$(git rev-parse --show-toplevel)"
BR=${BR:-/opt/code/wt-gc-target/release/intarsia}; MAIN=${MAIN:-/opt/code/wt-gc-main-target/release/intarsia}
M=/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4; M8=/Users/josef/Models/Qwen3-8B
G=$PWD/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
OR=$PWD/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
BD=golden/baselines; S35=qwen36-35b-a3b-nvfp4
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-23 05:30" +%s)
left() { echo $(( DEADLINE - $(date +%s) )); }
fits() { [ "$(left)" -gt $(( $1 * 60 )) ]; }   # fits MIN
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/job.log"; }
export INTARSIA_REPO=$PWD
log "start; left $(( $(left) / 60 )) min; branch $(git rev-parse --short HEAD)"
for b in "$BR" "$MAIN"; do [ -x "$b" ] || { log "missing binary $b"; exit 2; }; done

# 0. plan show (loads the model once, no decode)
for m in 8 35; do for t in main br; do
  eval B=\$$( [ $t = main ] && echo MAIN || echo BR ); eval MD=\$$( [ $m = 8 ] && echo M8 || echo M ); bud=40; [ $m = 8 ] && bud=26
  $B plan show --backend metal --model $MD --mem-budget $bud --max-seq 8192 --json < /dev/null > $O/plan-$m-$t.json 2> $O/plan-$m-$t.err
  echo "plan $m $t exit $?" >> $O/plan.txt
done; diff $O/plan-$m-main.json $O/plan-$m-br.json > $O/plan-$m.diff; echo "plan $m diff exit $? ($(wc -l < $O/plan-$m.diff) lines)" | tee -a $O/plan.txt; done

Q="qualify --backend metal --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 40 --stage-timeout-min 25 --model $M"
qual() { local n=$1; shift; local t0=$(date +%s)
  $BR $Q --out $O/q-$n "$@" < /dev/null > $O/q-$n.log 2>&1
  log "qualify $n exit $? $(( $(date +%s) - t0 )) s"
  grep -E "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" $O/q-$n.log | cut -c1-300 | tee -a $O/job.log; }

# 1. q-0: record at the new chunk (old chunk-256 baselines moved aside, not deleted)
if fits 45; then
  mkdir -p $O/old-baselines
  for f in $BD/$S35-vllm-top20.metal.baseline.json $BD/$S35-vllm-top20.metal.envelope.json $BD/$S35-vllm-4k-top20.metal.baseline.json $BD/$S35-cpu-oracle.metal.selfgolden.safetensors; do mv $f $O/old-baselines/; done
  qual 0-record --skip kernels --skip gate --skip bench --skip bench-long --skip ab --skip tp
  B0=$O/q-0-record/baselines; ls -l $B0 | tee -a $O/job.log
  cp $B0/acc-short.baseline.json $BD/$S35-vllm-top20.metal.baseline.json
  cp $B0/envelope.json $BD/$S35-vllm-top20.metal.envelope.json
  cp $B0/acc-4k.baseline.json $BD/$S35-vllm-4k-top20.metal.baseline.json
  cp $B0/*.selfgolden.safetensors $BD/
  git status --short $BD | tee -a $O/job.log
else log "SKIP q-0: does not fit"; fi

# 2. q-1: verdict against the new baselines
if fits 25; then qual 1-verdict --skip envelope --skip bench-long --skip ab --skip tp; else log "SKIP q-1"; fi

# 3. 256 greedy ids, main (chunk 256) vs branch (chunk 2048)
CORPUS=$PWD/tools/prompts/corpus_docs.txt
if fits 12; then
  for chars in 0 20000; do for t in main br; do
    eval B=\$$( [ $t = main ] && echo MAIN || echo BR )
    if [ $chars = 0 ]; then P="Write a short explanation of how a B-tree differs from a binary search tree."; else P="$(head -c $chars $CORPUS)"; fi
    $B generate --backend metal -m $M --mem-budget 40 --max-seq 8192 --profile-every 0 --ignore-eos --show-ids --max-tokens 256 -p "$P" > $O/gen-$chars-$t.out 2> $O/gen-$chars-$t.err < /dev/null
    grep -a '^ids:' $O/gen-$chars-$t.err > $O/gen-$chars-$t.ids
    log "gen $chars $t: $(grep -a 'prefill chunk' $O/gen-$chars-$t.err | head -1 | cut -c1-100) | $(grep -aEi 'prompt.*tokens' $O/gen-$chars-$t.err | head -1 | cut -c1-100) | ids md5 $(md5 -q $O/gen-$chars-$t.ids)"
  done; cmp -s $O/gen-$chars-main.ids $O/gen-$chars-br.ids && log "  $chars: ids identical" || log "  $chars: ids DIFFER ($(python3 - $O/gen-$chars-main.ids $O/gen-$chars-br.ids <<'P'
import sys,re
a,b=[re.findall(r'-?\d+',open(f).read()) for f in sys.argv[1:3]]
i=next((k for k,(x,y) in enumerate(zip(a,b)) if x!=y),None); print(f"first diff at index {i} of {len(a)}/{len(b)}")
P
))"; done
else log "SKIP ids"; fi

# 4. prefill tok/s at 4096, ABAB, 3 reps per server start
bench1() { local tag=$1 B=$2; local port=$((18080 + RANDOM % 500))
  $B serve --backend metal -m $M --mem-budget 40 --max-seq 6144 --host 127.0.0.1 --port $port --served-model-name g --profile-every 0 --no-supervisor --no-ui < /dev/null > $O/serve-$tag.log 2>&1 &
  local sp=$!
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:$port/v1/models > /dev/null && break; sleep 2; done
  $BR bench speed --url http://127.0.0.1:$port --model g --sizes 4096 --gen 16 --reps 3 --warmup 1 --thinking off --no-save --json > $O/bench-$tag.json 2> $O/bench-$tag.err < /dev/null
  grep -a "prefill chunk" $O/serve-$tag.log | head -1 | tee -a $O/job.log
  kill $sp; wait $sp 2>/dev/null; log "bench $tag done (rc $?)"; sleep 3; }
if fits 15; then for r in 1 2; do bench1 main$r $MAIN; bench1 br$r $BR; done; else log "SKIP tok/s"; fi
log "done; left $(( $(left) / 60 )) min"
