#!/usr/bin/env bash
# Lane P1 (Qwen3.8-27B MXFP4 W4A4 decode on HIP): GPU stages. Runs ON tatooine inside a fleetq lease with radiance
# down (scripts/p1/burst.sh stops radiance, runs this, restarts radiance and waits for it). One R9700 (RS_GPU, default 0).
#   scripts/p1/remote.sh prof:TAG:CTX optable:TAG:CTX bench:TAG acc:TAG kernels ...
# CTX = prompt size in tokens (short|1k|4k). Needs the release build (scripts/p1/build.sh, CPU slot, no lease).
# Profiling runs in the derived image intarsia-hip-prof against the same cargo target volume as the dev image.
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-p1}" HIP_VISIBLE_DEVICES="${RS_GPU:-0}"
export INTARSIA_TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-p1}"
export INTARSIA_MODELS=/data/models/radiance/hf-cache/hub
cd "$INTARSIA_REPO"
D=deploy/tatooine/dev.sh
B=/target/release/intarsia
M=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
G=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-top20.safetensors
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
SERVE=p1-serve
GB=15.49   # weight GB read per decoded token: 12.94 MXFP4 text layers + 2.54 bf16 lm_head (safetensors headers)
OUTD=${OUTD:-$INTARSIA_REPO/bench-out}
ts() { date +%T; }
trap 'docker rm -f $SERVE >/dev/null 2>&1' EXIT
mkdir -p "$OUTD"

prompt_for() { # prompt_for <ctx> -> path (inside /work) of a prompt file of about that many tokens
  case "$1" in
    short) echo "Write a long story about a robot who learns to paint." > "$OUTD/p-short.txt" ;;
    1k) head -c 3700 scripts/hip-decode/prompt4k.txt > "$OUTD/p-1k.txt" ;;
    4k) cp scripts/hip-decode/prompt4k.txt "$OUTD/p-4k.txt" ;;
  esac
  echo "/work/bench-out/p-$1.txt"
}

gen_script() { # gen_script <ctx> <extra> -> writes bench-out/gen.sh (a generate of 128 decode tokens)
  local p; p=$(prompt_for "$1")
  echo "$B generate -m $M --backend hip --mem-budget 30 --max-seq 8192 --profile-every ${PE:-0} --ignore-eos $2 -p \"\$(cat $p)\" --max-tokens ${NTOK:-128}" > "$OUTD/gen.sh"
}

stage_prof() { # prof:<tag>:<ctx>: unprofiled rate + rocprofv3 kernel trace of a 128-token decode after a <ctx> prompt
  local tag=${1%%:*} ctx=${1#*:}
  [ "$ctx" = "$1" ] && ctx=short
  gen_script "$ctx" "${PROF_EXTRA:-}"
  echo "--- unprofiled ($ctx) [$(ts)]"
  $D bash -c "bash /work/bench-out/gen.sh" < /dev/null 2>&1 | grep -aE "prompt:|finish=|steps=|WARNING|graph"
  echo "--- rocprofv3 ($ctx) [$(ts)]"
  local d="$OUTD/prof/$tag-$ctx"
  rm -rf "$d"; mkdir -p "$d"
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 \
    $D bash -c "timeout -s KILL ${PROF_TIMEOUT:-400} rocprofv3 --kernel-trace --output-format csv -d /work/bench-out/prof/$tag-$ctx -o run -- bash /work/bench-out/gen.sh" < /dev/null 2>&1 | grep -aE "finish=|steps=|rror"
  local csv; csv=$(find "$d" -name '*kernel_trace.csv' | head -1)
  [ -n "$csv" ] && python3 scripts/hip-decode/analyze.py "$csv" > "$d/table.txt" && head -70 "$d/table.txt"
}

stage_optable() { # optable:<tag>:<ctx>: eager decode, every step profiled, per-Op table
  local tag=${1%%:*} ctx=${1#*:}
  echo "--- op-table ($ctx, eager, profile every step) [$(ts)]"
  PE=1 gen_script "$ctx" "--no-graphs --op-table"
  $D bash -c "bash /work/bench-out/gen.sh" < /dev/null > "$OUTD/optable-$tag-$ctx.txt" 2>&1
  grep -aE "prompt:|finish=|steps=" "$OUTD/optable-$tag-$ctx.txt"
  grep -avE "^\s*$" "$OUTD/optable-$tag-$ctx.txt" | tail -45
}

stage_bench() { # bench:<tag>: serve (graphs, default knobs + SERVE_ENV) and bench speed at SIZES
  local tag=$1 port=8091
  docker rm -f $SERVE >/dev/null 2>&1
  echo "--- serve $tag [$(ts)]"
  docker run -d --name $SERVE --network host \
    --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --group-add video --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES ${SERVE_ENV:-} \
    -v "$INTARSIA_REPO":/work -v "$INTARSIA_MODELS":/models:ro -v "$INTARSIA_TARGET_VOL":/target -w /work \
    $IMG $B serve --backend hip --model $M --profile-every 0 --mem-budget 30 --host 0.0.0.0 --port $port \
    --served-model-name qwen38-27b-p1 --max-seq ${MAXSEQ:-8192} --no-ui < /dev/null > /dev/null
  local up=0 i
  for i in $(seq 1 200); do
    curl -sf -m 2 http://127.0.0.1:$port/health > /dev/null 2>&1 && { up=1; break; }
    docker ps -q -f name=$SERVE | grep -q . || break
    sleep 3
  done
  [ "$up" != 1 ] && { echo "serve did not come up"; docker logs $SERVE 2>&1 | tail -30; docker rm -f $SERVE > /dev/null; return 1; }
  echo "up after ~$((i*3))s"
  docker exec $SERVE $B bench speed --url http://127.0.0.1:$port --model qwen38-27b-p1 \
    --sizes ${SIZES:-1024,4096} --gen ${GEN:-128} --reps ${REPS:-3} --gb-per-token $GB --bandwidth-gbs 627 --at-most \
    --label "p1-27b-$tag" --out-dir /work/bench-out --thinking off < /dev/null 2>&1 | tail -14
  curl -s http://127.0.0.1:$port/metrics | grep -iE "graph|eager" | head -6
  docker rm -f $SERVE > /dev/null
}

stage_acc() { # acc:<tag>: 27B W4A4 golden, prefill + decode, ref-noise sidecar (found by the golden's name)
  local tag=${1:-base}
  mkdir -p "$OUTD/acc"
  echo "--- accuracy 27B W4A4 ($tag) [$(ts)]"
  $D $B accuracy --backend hip --model $M --mem-budget 30 --golden $G --mode both \
    --ref-noise /work/golden/ref_noise/qwen3.8-27b-mxfp4-hfquark-top20.safetensors.json < /dev/null > "$OUTD/acc/acc27-$tag.log" 2>&1
  grep -aE "mode|KL|top-1|PASS|FAIL|noise|verdict" "$OUTD/acc/acc27-$tag.log" | tail -30
}

stage_kernels() { # kernels test (all HIP kernels; 27B model shapes)
  echo "--- kernels test [$(ts)]"
  $D $B kernels test --backend hip --model $M < /dev/null > "$OUTD/kernels-test.log" 2>&1
  grep -aE "FAIL|REJECT|kernels,|mxfp4" "$OUTD/kernels-test.log" | tail -30
}

ids_run() { # ids_run <file tag> <models dir (host)> <model (container)> <ctx> [extra]: 256 greedy tokens -> bench-out/ids/<tag>.ids
  local tag=$1 mdir=$2 model=$3 ctx=$4; shift 4
  mkdir -p "$OUTD/ids"
  local p; p=$(prompt_for "$ctx")
  INTARSIA_MODELS=$mdir $D env ${GEN_ENV:-X=0} $B generate -m "$model" --backend hip --mem-budget 29 --max-seq 8192 \
    --profile-every 0 --ignore-eos --show-ids -p "$(cat "$INTARSIA_REPO/${p#/work/}")" --max-tokens 256 "$@" < /dev/null \
    > "$OUTD/ids/$tag.out" 2> "$OUTD/ids/$tag.err"
  grep -a '^ids:' "$OUTD/ids/$tag.err" > "$OUTD/ids/$tag.ids"
  echo "$tag: $(md5sum < "$OUTD/ids/$tag.ids" | cut -c1-8) $(grep -aE 'finish=' "$OUTD/ids/$tag.err" | head -1 | cut -c1-160)"
}
MI=/data/models/intarsia
stage_ids27() { local c; for c in short 4k; do ids_run "27-${1:-base}-$c" "$INTARSIA_MODELS" "$M" $c; done; }
stage_ids35() { local c; for c in short 4k; do ids_run "35-${1:-base}-$c" $MI /models/Qwen3.6-35B-A3B-NVFP4 $c; done; }
stage_ids8() { local c; for c in short 4k; do ids_run "8-${1:-base}-$c" $MI /models/Qwen3-8B $c; done; }

stage_rate() { # rate:<ctx>:<KNOB=V,...>: one unprofiled 128-token decode after a <ctx> prompt with the given env
  local ctx=${1%%:*} kv=${1#*:}
  [ "$kv" = "$1" ] && kv=X=0
  gen_script "$ctx" ""
  echo "rate $ctx $kv: $($D env ${kv//,/ } bash /work/bench-out/gen.sh < /dev/null 2>&1 | grep -aE 'finish=' | head -1 | grep -oE 'decode [0-9]+ ms \([0-9.]+ tok/s\)')"
}

BASEVOL=intarsia-cargo-target-p1base
NEWVOL=intarsia-cargo-target-p1
stage_ab() { # ab:<n>: interleaved A (p1base vol) / B (p1 vol) bench speed, n rounds, 1 rep each (+1 warmup)
  local i
  for i in $(seq 1 "${1:-3}"); do
    INTARSIA_TARGET_VOL=$BASEVOL REPS=1 stage_bench "A$i"
    INTARSIA_TARGET_VOL=$NEWVOL REPS=1 stage_bench "B$i"
  done
}
stage_bench32k() { # the drill's serve shape (--max-seq 32768): do graphs still replay?
  INTARSIA_TARGET_VOL=$BASEVOL MAXSEQ=32768 SIZES=1024 REPS=1 stage_bench "base-ms32k"
  docker logs $SERVE 2>&1 | grep -iE "graph" | head -5
}
stage_selfgolden() { # base build writes its own logits on the golden's prompts; the new build is compared (KL 0 = bitwise)
  mkdir -p "$OUTD/acc"
  INTARSIA_TARGET_VOL=$BASEVOL $D $B accuracy --backend hip --model $M --mem-budget 30 --golden $G --mode both \
    --write-self-golden /work/bench-out/acc/sg27-base.safetensors --allow-unstamped < /dev/null > "$OUTD/acc/sg27-write.log" 2>&1
  tail -3 "$OUTD/acc/sg27-write.log"
  INTARSIA_TARGET_VOL=$NEWVOL $D $B accuracy --backend hip --model $M --mem-budget 30 --mode both \
    --self-golden /work/bench-out/acc/sg27-base.safetensors --allow-unstamped < /dev/null > "$OUTD/acc/sg27-check.log" 2>&1
  grep -aiE "KL|flip|top|PASS|FAIL|verdict" "$OUTD/acc/sg27-check.log" | tail -15
}
stage_idsab() { # 35B + 8B greedy ids, base vs new (must be identical)
  local v
  for v in base new; do
    local vol=$BASEVOL; [ $v = new ] && vol=$NEWVOL
    INTARSIA_TARGET_VOL=$vol stage_ids35 $v
    INTARSIA_TARGET_VOL=$vol stage_ids8 $v
  done
  local t
  for t in 35-short 35-4k 8-short 8-4k; do
    local m=${t%%-*} c=${t#*-}
    cmp -s "$OUTD/ids/$m-base-$c.ids" "$OUTD/ids/$m-new-$c.ids" && [ -s "$OUTD/ids/$m-base-$c.ids" ] && echo "IDS SAME $t" || echo "IDS DIFFER $t"
  done
}

for s in "$@"; do
  name=${s%%:*}; arg=${s#*:}; [ "$arg" = "$s" ] && arg=""
  echo "=== stage $s [$(ts)]"
  "stage_$name" $arg
done
