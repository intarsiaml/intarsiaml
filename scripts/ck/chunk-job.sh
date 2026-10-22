#!/usr/bin/env bash
# Lane ck: does --prefill-chunk 4096 / 8192 change prefill numerics, memory or speed vs the 2048 default? Runs ON the box,
# inside an EXCLUSIVE fleetq slot (speed numbers), one backend per call:
#   chunk-job.sh cuda|metal|hip <outdir> [steps: acc plan speed]
# Per model (35B NVFP4, Qwen3-8B) and chunk C in $CHUNKS (default 2048 4096 8192):
#   gen    greedy ids + top-20 logits on 5.8k / 16k-token corpus prompts (round 6's chunk-check.sh), identical y/n
#   accdec the same dumps in decode mode (teacher-forced single-token steps after the chunked prefill)
#   acc    `intarsia accuracy --mode prefill --write-self-golden` on the short and 4k goldens plus one synthetic 16000-token prompt (scripts/ck/make_synth.py); scripts/ck/chunkcmp.py then compares
#          the three dumps row by row, bit by bit (the 2048 dump is the base)
#   speed  a FRESH `serve --prefill-chunk C` per chunk (order 2048 4096 8192 2048, the repeat is a drift control); its
#          GET /v1/intarsia/plan is saved (memory plan) and `bench speed --at-most --sizes 4096,16384 --reps 3`
# cuda: binary ~/ck/target/release/intarsia (scripts/ck/cuda-build-spark1.sh), goldens ~/ck/golden.  metal: local build in
# /opt/code/wt-ck-target, goldens from the scratchpad.  hip: ~/ck-hip tree in the dev image (deploy/tatooine/dev.sh), needs radiance down.
set -uo pipefail
BE=${1:?cuda|metal|hip}; OUT=${2:?outdir}; STEPS=${3:-acc speed}
CHUNKS=${CHUNKS:-2048 4096 8192}
MAXSEQ=${MAXSEQ:-17408}   # 16k prompts + 16 generated tokens; the HIP recipe fits the 29 GB budget at this length
MODELS_RUN=${MODELS_RUN:-35b 8b}
mkdir -p "$OUT"
want() { [[ " $STEPS " == *" $1 "* ]]; }
ts() { date +%T; }
PORT=${PORT:-8093}; NAME=ck-serve; SRV=ck-m
case $BE in
  cuda)
    BIN=${BIN:-$HOME/ck/target/release/intarsia}
    M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
    M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
    GD=${GD:-$HOME/ck/golden}; OUTIN=$OUT; B35=${B35:-40}; B8=${B8:-26}
    CORPUS=$HOME/ck/src/tools/prompts/corpus_docs.txt
    run() { "$BIN" "$@"; }
    srv_start() { nohup "$BIN" serve --backend cuda "$@" --host 127.0.0.1 --port $PORT --served-model-name $SRV --profile-every 0 < /dev/null > "$SRVLOG" 2>&1 & SRVPID=$!; }
    srv_stop() { kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; }
    srv_alive() { kill -0 "$SRVPID" 2>/dev/null; }
    bench() { "$BIN" bench speed "$@"; }
    ;;
  metal)
    BIN=${BIN:-/opt/code/wt-ck-target/release/intarsia}
    M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M8=$HOME/models/Qwen3-8B
    GD=${GD:-/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/ck/gold}; OUTIN=$OUT; B35=${B35:-40}; B8=${B8:-26}
    CORPUS=${CORPUS:-/opt/code/wt-ck/tools/prompts/corpus_docs.txt}
    run() { "$BIN" "$@"; }
    srv_start() { nohup "$BIN" serve --backend metal "$@" --host 127.0.0.1 --port $PORT --served-model-name $SRV --profile-every 0 < /dev/null > "$SRVLOG" 2>&1 & SRVPID=$!; }
    srv_stop() { kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; }
    srv_alive() { kill -0 "$SRVPID" 2>/dev/null; }
    bench() { "$BIN" bench speed "$@"; }
    ;;
  hip)
    REPO=${INTARSIA_REPO:-$HOME/ck-hip}
    export INTARSIA_REPO=$REPO INTARSIA_TARGET_VOL=${INTARSIA_TARGET_VOL:-intarsia-cargo-target-ck} HIP_VISIBLE_DEVICES=${RS_GPU:-0}
    CORPUS=$REPO/tools/prompts/corpus_docs.txt
    D=$REPO/deploy/tatooine/dev.sh; IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0; BIN=/target/release/intarsia
    M35=/models/Qwen3.6-35B-A3B-NVFP4; M8=/models/Qwen3-8B
    GD=/work/ck-data; OUTIN=/work/${OUT#$REPO/}; B35=${B35:-29}; B8=${B8:-29}
    run() { (cd "$REPO" && "$D" "$BIN" "$@"); }
    srv_start() {
      docker rm -f $NAME >/dev/null 2>&1
      docker run -d --name $NAME --network host --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" \
        --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video --ipc host --security-opt seccomp=unconfined \
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
        -v "$REPO":/work -v /data/models/intarsia:/models:ro -v "$("$D" --target-vol)":/target -w /work \
        $IMG $BIN serve --backend hip "$@" --host 0.0.0.0 --port $PORT --served-model-name $SRV --profile-every 0 > /dev/null
    }
    srv_stop() { docker logs $NAME > "$SRVLOG" 2>&1; docker rm -f $NAME >/dev/null 2>&1; }
    srv_alive() { docker ps -q -f name=$NAME | grep -q .; }
    bench() { docker exec $NAME $BIN bench speed "$@"; }
    trap 'docker rm -f $NAME >/dev/null 2>&1' EXIT
    ;;
  *) echo "backend?"; exit 2 ;;
esac
# model id -> path, budget, goldens
mpath() { [[ $1 == 35b ]] && echo "$M35" || echo "$M8"; }
mbud() { [[ $1 == 35b ]] && echo "$B35" || echo "$B8"; }
goldens() { if [[ $1 == 35b ]]; then echo "all:$GD/qwen36-35b-s16k.safetensors"; else echo "short:$GD/qwen3-8b-short.safetensors 4k:$GD/qwen3-8b-4k-s16k.safetensors"; fi; }
gclass() { [[ $1 == 35b ]] && echo 4bit || echo bf16; }
echo "== chunk-job $BE start $(ts) host $(hostname) steps '$STEPS' chunks '$CHUNKS' job ${FLEETQ_JOB:-none}" | tee -a "$OUT/steps.log"
if [[ $BE == cuda ]]; then free -g | sed -n 2p | tee -a "$OUT/steps.log"; nvidia-smi --query-gpu=clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader | tee -a "$OUT/steps.log"; fi

acc_run() { # acc_run <mode> <step name>: self-golden dumps of every (model, golden, chunk)
  local mode=$1 ok=$2 m g gn gf d c t0
  want "$ok" || return 0
  for m in $MODELS_RUN; do
    for g in $(goldens $m); do gn=${g%%:*}; gf=${g#*:}
      d=$OUT/acc/$m-$gn; mkdir -p "$d"
      for c in $CHUNKS; do
        t0=$(date +%s)
        run accuracy --backend $BE --model "$(mpath $m)" --golden "$gf" --mode $mode --prefill-chunk "$c" --max-seq "$MAXSEQ" \
          --mem-budget "$(mbud $m)" --gate-class "$(gclass $m)" --allow-unstamped --allow-dirty \
          --write-self-golden "${OUTIN}/acc/$m-$gn/$mode-c$c.safetensors" > "$d/$mode-c$c.log" 2>&1 < /dev/null
        echo "$ok $m $gn $mode chunk $c exit $? $(( $(date +%s) - t0 )) s $(ls -la "$d/$mode-c$c.safetensors" 2>/dev/null | awk '{print $5}') bytes" | tee -a "$OUT/steps.log"
      done
    done
  done
}

acc_run prefill acc
if want gen; then
  # round 6's chunk-check.sh: 64 greedy ids + top-20 logits of the first generated token on 5.8k / 16k-token prompts, per chunk
  for m in $MODELS_RUN; do
    mkdir -p "$OUT/gen/$m"
    for chars in ${GEN_CHARS:-20000 54000}; do
      for c in $CHUNKS; do
        t0=$(date +%s); f=$OUT/gen/$m/gen-$chars-c$c
        run generate --backend $BE -m "$(mpath $m)" --mem-budget "$(mbud $m)" --max-seq "$MAXSEQ" --prefill-chunk "$c" --profile-every 0 \
          --ignore-eos --show-ids --show-logits 20 --max-tokens ${GEN:-64} -p "$(head -c $chars "$CORPUS")" > $f.out 2> $f.err < /dev/null
        rc=$?; grep -a '^ids:' $f.err > $f.ids; grep -a '^top logits' $f.err > $f.top
        echo "gen $m $chars chars chunk $c exit $rc $(( $(date +%s) - t0 )) s: $(grep -aE 'finish=' $f.err | head -1 | cut -c1-120) ids $(md5sum < $f.ids | cut -c1-8) top $(md5sum < $f.top | cut -c1-8)" | tee -a "$OUT/steps.log"
      done
      first=${CHUNKS%% *}
      for c in $CHUNKS; do [[ $c == "$first" ]] && continue
        i=$(cmp -s $OUT/gen/$m/gen-$chars-c$first.ids $OUT/gen/$m/gen-$chars-c$c.ids && echo identical || echo DIFFER)
        l=$(cmp -s $OUT/gen/$m/gen-$chars-c$first.top $OUT/gen/$m/gen-$chars-c$c.top && echo identical || echo DIFFER)
        echo "  gen $m $chars chars: chunk $c vs $first: ids $i, first-token top-20 logits $l" | tee -a "$OUT/steps.log"
      done
    done
  done
fi

if want speed || want plan; then
  for m in $MODELS_RUN; do
    mkdir -p "$OUT/speed/$m"
    for c in ${SPEED_CHUNKS:-$CHUNKS 2048}; do
      SRVLOG=$OUT/speed/$m/serve-c$c-$(date +%H%M%S).log
      echo "--- $m chunk $c serve [$(ts)]" | tee -a "$OUT/steps.log"
      srv_start --model "$(mpath $m)" --prefill-chunk "$c" --max-seq "$MAXSEQ" --mem-budget "$(mbud $m)"
      up=0
      for i in $(seq 1 240); do
        curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }
        srv_alive || break
        sleep 3
      done
      if [[ $up != 1 ]]; then echo "serve $m chunk $c DID NOT COME UP (see $SRVLOG)" | tee -a "$OUT/steps.log"; srv_stop; continue; fi
      echo "up after ~$((i * 3)) s" | tee -a "$OUT/steps.log"
      curl -sf -m 10 http://127.0.0.1:$PORT/v1/intarsia/plan > "$OUT/speed/$m/plan-c$c-$(date +%H%M%S).json" 2>&1
      # memory: the plan (scratch_bytes) above, plus the server's own gauges right after load and after the bench
      curl -sf -m 10 http://127.0.0.1:$PORT/metrics 2>/dev/null | grep -E "^intarsia_(device_memory|kv_bytes_total|weights_bytes)" > "$OUT/speed/$m/mem-c$c-load.txt"
      [[ $BE == cuda ]] && { free -m | sed -n 2p; } >> "$OUT/speed/$m/mem-c$c-load.txt"
      if want speed; then
        bench --url http://127.0.0.1:$PORT --model $SRV --sizes 4096,16384 --gen 16 --reps 3 --warmup 1 --at-most \
          --prefill-chunk "$c" --label "ck-$BE-$m-c$c" --out-dir "$OUTIN/speed/$m" --thinking off 2>&1 | tail -12 | tee -a "$OUT/speed/$m/bench-c$c.txt" "$OUT/steps.log"
      fi
      curl -sf -m 10 http://127.0.0.1:$PORT/metrics 2>/dev/null | grep -E "^intarsia_(device_memory|kv_bytes_total|weights_bytes)" > "$OUT/speed/$m/mem-c$c-after.txt"
      [[ $BE == cuda ]] && { free -m | sed -n 2p; } >> "$OUT/speed/$m/mem-c$c-after.txt"
      srv_stop
    done
  done
fi
acc_run decode accdec
echo "== chunk-job $BE done $(ts)" | tee -a "$OUT/steps.log"
