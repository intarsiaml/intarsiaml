#!/usr/bin/env bash
# Lane G1b (docs/design/35b-tp2.md): the 35B tp=2 accuracy gates on ONE GB10, both ranks on device 0 joined by the
# host communicator. A two-operand sum commutes, so the NCCL sum is the host comm's bit for bit: these are the
# cross-host numerics (checked by a greedy-ids diff in the cross-host smoke). vLLM stays UP; non-exclusive slot:
#   PRE_CHANGE=<origin/main sha> fleetq run --host spark2 --mem 45 --label "G1b gates" -- scripts/35b-tp2/gates-onebox.sh
# STEPS (space separated) picks steps; default all. Output: $HOST:~/intarsia-tp2/out/$OUT, copied to
# docs/results/2026-10-21-35b-tp2-cuda/raw/$OUT. One model process at a time.
#
# Self-golden: S1 is the BASE build (origin/main) at tp=1. dev tp=1 against it must be exactly zero (end-to-end tp=1
# identity); dev tp=2 against it is the TP gate; a tp=1 SIMT-attention variant against it is the 35B's own reordering.
# TP budget = docs/accuracy.md "TP self-golden budget" derived row (8B): mean KL 1.2e-3, max position KL 9.8e-2,
# flips 0.84 %, tail 4.2e-4, worst prompt 1.92 x mean.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
: "${PRE_CHANGE:?PRE_CHANGE=<origin/main sha the base binary was built from>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
OUT=${OUT:-gates2}
HOST=${HOST:-spark2}
LOCAL=$REPO/docs/results/2026-10-21-35b-tp2-cuda/raw/$OUT
mkdir -p "$LOCAL"
cat > "$LOCAL/remote.sh" <<'EOS'
set -u
P=$HOME/intarsia-tp2; O=$P/out/$OUT; mkdir -p $O
M35=$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
SG=$P/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; LG=$P/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
CORPUS=$P/prompts/corpus_docs.txt
BASE=$P/bin/base-intarsia; DEV=$P/bin/dev-intarsia
TP2="--tp 2 --comm host --devices 0,0"
TPB="--self-max-kl 1.2e-3 --self-max-pos-kl 9.8e-2 --self-max-flip-pct 0.84 --self-max-tail 4.2e-4 --self-prompt-factor 1.92"
PC="--pre-change $PRE_CHANGE"
SIMT=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash
COMMON="--backend cuda --max-seq 16384 --prefill-chunk 2048"
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); log "MemAvailable ${gb} GB"; (( gb >= 42 )) || { log "ABORT: memory"; exit 1; }
for b in $BASE $DEV; do log "bin $b: $($b --version 2>&1 | tr '\n' ' ')"; done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader | tee -a $O/steps.log
plan() { # plan <tag> <bin> <model> <budget> <extra>
  want plan-$1 || return 0
  timeout 600 $2 plan show --backend cuda --model $3 --max-seq 4096 --mem-budget $4 $5 --json > $O/plan-$1.json 2> $O/plan-$1.err
  log "plan $1 exit $? $(wc -c < $O/plan-$1.json) bytes"
}
plan base-35b $BASE $M35 29 ""
plan dev-35b $DEV $M35 29 ""
plan base-8b $BASE $M8 25 ""
plan dev-8b $DEV $M8 25 ""
plan dev-35b-tp2 $DEV $M35 18 "$TP2"
if want plan-base-35b && want plan-dev-35b; then
  cmp -s $O/plan-base-35b.json $O/plan-dev-35b.json && log "PLAN IDENTITY 35b tp=1: identical" || log "PLAN IDENTITY 35b tp=1: DIFFERS"
  cmp -s $O/plan-base-8b.json $O/plan-dev-8b.json && log "PLAN IDENTITY 8b tp=1: identical" || log "PLAN IDENTITY 8b tp=1: DIFFERS"
fi
acc() { # acc <tag> <bin> <golden> <budget> <args...>
  local tag=$1 bin=$2 g=$3 mb=$4; shift 4; want acc-$tag || return 0; local t0=$(date +%s)
  timeout 1800 $bin accuracy --model $M35 $COMMON --golden $g --mem-budget $mb "$@" > $O/acc-$tag.log 2>&1
  log "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE '^prefill floor|^decode floor|verdict' $O/acc-$tag.log | head -3 | cut -c1-170 | tr '\n' '|')"
}
selfg() { # selfg <tag> <self golden> <budget> <args...>
  local tag=$1 g=$2 mb=$3; shift 3; want self-$tag || return 0; local t0=$(date +%s)
  timeout 1800 $DEV accuracy --model $M35 $COMMON --self-golden $g --mode both --mem-budget $mb --allow-config-mismatch "$@" > $O/self-$tag.log 2>&1
  log "self $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE 'SELF-STATS|self-golden gate|^Error' $O/self-$tag.log | cut -c1-260 | tr '\n' '|')"
}
acc base-tp1-short $BASE $SG 40 --mode both --write-self-golden $O/S1-short.safetensors
acc base-tp1-long $BASE $LG 40 --mode both --write-self-golden $O/S1-long.safetensors
acc tp2-short $DEV $SG 18 --mode both $TP2
acc tp2-long $DEV $LG 18 --mode both $TP2
selfg tp1-short $O/S1-short.safetensors 40 $PC
selfg tp1-long $O/S1-long.safetensors 40 $PC
selfg tp2-short $O/S1-short.safetensors 18 $TP2 $TPB $PC
selfg tp2-long $O/S1-long.safetensors 18 $TP2 $TPB $PC
INTARSIA_CUDA_DISABLE=$SIMT selfg tp1simt-short $O/S1-short.safetensors 40 $TPB $PC
INTARSIA_CUDA_DISABLE=$SIMT selfg tp1simt-long $O/S1-long.safetensors 40 $TPB $PC
gen() { # gen <tag> <chars> <tokens> <args...>; chat template, greedy
  local tag=$1 chars=$2 n=$3; shift 3; want gen-$tag || return 0
  timeout 900 $DEV generate -m $M35 --backend cuda --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c $chars $CORPUS)" --max-tokens $n "$@" > $O/gen-$tag.out 2> $O/gen-$tag.err
  grep -a '^ids:' $O/gen-$tag.err > $O/gen-$tag.ids
  log "gen $tag: $(grep -aE 'finish=' $O/gen-$tag.err | head -1 | cut -c1-160)"
}
gen tp1-5k 5000 256 --mem-budget 29
INTARSIA_CUDA_DISABLE=$SIMT gen tp1simt-5k 5000 256 --mem-budget 29
gen tp2-5k 5000 256 --mem-budget 16 $TP2
gen tp1-16k 54000 256 --mem-budget 29
INTARSIA_CUDA_DISABLE=$SIMT gen tp1simt-16k 54000 256 --mem-budget 29
gen tp2-16k 54000 256 --mem-budget 16 $TP2
reuse() { # reuse <tag> <budget> <args...>
  local tag=$1 mb=$2; shift 2; want reuse-$tag || return 0; local t0=$(date +%s)
  timeout 1800 $DEV accuracy --model $M35 $COMMON --golden $LG --mode reuse --mem-budget $mb "$@" > $O/reuse-$tag.log 2>&1
  log "reuse $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE '^chat_reuse_4k|prefix-reuse gate' $O/reuse-$tag.log | cut -c1-120 | tr -s ' ' | tr '\n' '|')"
}
reuse tp2 18 $TP2
python3 - $O <<'PY'
import sys, re
o = sys.argv[1]
def ids(t):
    try:
        s = open(f"{o}/gen-{t}.ids").read().split(":", 1)[1]
        return [int(v) for v in re.findall(r"-?\d+", s)]
    except Exception:
        return None
for a, b in [("tp1-5k", "tp2-5k"), ("tp1-5k", "tp1simt-5k"), ("tp1-16k", "tp2-16k"), ("tp1-16k", "tp1simt-16k")]:
    x, y = ids(a), ids(b)
    if x is None or y is None:
        print(f"IDS {a} vs {b}: missing"); continue
    d = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    print(f"IDS {a} vs {b}: {len(x)} / {len(y)} ids, first divergence at {d if d is not None else 'none'}")
PY
echo "== done $(date +%T)"
EOS
ssh -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB OUT=$OUT STEPS='${STEPS:-}' PRE_CHANGE=$PRE_CHANGE; bash -s" < "$LOCAL/remote.sh" 2>&1 | tee "$LOCAL/console.log"
rsync -a --exclude '*.safetensors' "$HOST:intarsia-tp2/out/$OUT/" "$LOCAL/"
