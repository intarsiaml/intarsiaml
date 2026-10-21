#!/usr/bin/env bash
# Lane G1c: CUDA graphs for the tp=2 decode step, 35B across the two Sparks (rank 0 = spark1 with `--tp-remote`, rank 1 =
# spark2 `intarsia tp-worker`, NCCL 2.30.7 over ConnectX-7 port f1). Derived from scripts/35b-tp2/xhost.sh. Runs HERE (the
# Mac) as the one command of ONE fleetq job that holds both Sparks:
#   gates (vLLM up):   MODE=gates  fleetq run --host spark1,spark2 --mem spark1=30,spark2=20 --label "..." -- scripts/tp2-graphs/xg.sh
#   mutants (vLLM up): MODE=mut    (same slot shape; MUTANTS="name ...", binaries g1c-mut-<name>)
#   speed (vLLM down): MODE=window fleetq run --host spark1,spark2 --exclusive --takes-down flight,guidance,capcom,retro \
#                        --mem spark1=100,spark2=95 --label "..." -- scripts/tp2-graphs/xg.sh   (no --each: one driver)
# The window stops both vLLM containers and ALWAYS restarts them (trap on EXIT: docker start, wait for /v1/models), kills
# every process it started, and appends one line per role to raw/downtime.log. STEPS picks steps (default all).
# Binaries: ~/intarsia-tp2/bin/g1c-$TREE-intarsia on both boxes (TREE default dev; BASE default base). Output: raw/$OUT.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
MODE=${MODE:?gates, mut or window}
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-21-tp2-cuda-graphs/raw
OUT=${OUT:-$MODE-$(date +%H%M)}
L=$RAW/$OUT; mkdir -p "$L"
TREE=${TREE:-dev}; BASE=${BASE:-base}
MAX_MIN=${MAX_MIN:-24}   # window work budget (the vLLM restart comes on top; the cap is 30 min down)
DEADLINE=$(( $(date +%s) + MAX_MIN * 60 ))
BIN="\$HOME/intarsia-tp2/bin/g1c-$TREE-intarsia"; BBIN="\$HOME/intarsia-tp2/bin/g1c-$BASE-intarsia"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
WIP=10.100.152.3; WPORT=29555
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
VLLM_C=missioncntrl-guidance
vllm_url() { [[ $1 == spark1 ]] && echo http://100.120.88.104:8000/v1/models || echo http://100.96.95.89:8000/v1/models; } # bash 3.2: no -A
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date +%T) $*" | tee -a "$L/steps.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
r() { local h=$1; shift; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }

cleanup_procs() {
  for h in spark1 spark2; do
    r $h "touch ~/intarsia-tp2/stop-worker; pkill -TERM -f '[i]ntarsia-tp2/bin/g1c-'; pkill -TERM -f '[n]sys profile'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-tp2/bin/g1c-' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-tp2/bin/g1c-'; true"
  done
}
STOPPED=0; T_STOP=0
restore() {
  local rc=$?
  log "restore (exit $rc)"
  cleanup_procs
  if [[ $STOPPED == 1 ]]; then
    for h in spark1 spark2; do r $h "docker start $VLLM_C >/dev/null" || log "WARNING: docker start on $h failed"; done
    for h in spark1 spark2; do
      local ok=0
      for i in $(seq 1 240); do curl -sf -m 5 "$(vllm_url $h)" >/dev/null && { ok=1; break; }; sleep 5; done
      local t=$(date +%s)
      [[ $ok == 1 ]] && log "vLLM on $h serving again" || log "WARNING: vLLM on $h did not answer /v1/models within 20 min"
      local role=$([[ $h == spark1 ]] && echo spark1-vllm || echo spark2-vllm)
      printf '%s job=%s role=%s stop=%s up=%s down=%d s label="G1c tp2 graphs %s" serving=%s\n' "$(date +%F)" "$FLEETQ_JOB" "$role" \
        "$(date -r $T_STOP +%H:%M:%S)" "$(date -r $t +%H:%M:%S)" $(( t - T_STOP )) "$OUT" "$([[ $ok == 1 ]] && echo yes || echo NO)" >> "$RAW/downtime.log"
    done
  fi
  rsync -a "spark1:intarsia-tp2/xout/$OUT/" "$L/" 2>/dev/null; rsync -a "spark2:intarsia-tp2/xout/$OUT/" "$L/spark2/" 2>/dev/null
  trap - EXIT INT TERM
  exit $rc
}
trap restore EXIT INT TERM

for h in spark1 spark2; do r $h "mkdir -p ~/intarsia-tp2/xout/$OUT; rm -f ~/intarsia-tp2/stop-worker; echo $h: \$($BIN --version | tr '\n' ' ')" | tee -a "$L/steps.log"; done
if [[ $MODE == window ]]; then
  T_STOP=$(date +%s); STOPPED=1
  for h in spark1 spark2; do r $h "docker stop -t 30 $VLLM_C >/dev/null" && log "vLLM stopped on $h"; done
  for h in spark1 spark2; do
    for i in $(seq 1 30); do gb=$(r $h "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo"); (( gb >= 95 )) && break; sleep 2; done
    log "$h MemAvailable $gb GB"
  done
fi
r spark1 "nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader" | tee -a "$L/steps.log"

XO='~/intarsia-tp2/xout/'$OUT
GD='$HOME/intarsia-tp2/golden'; VG='$HOME/intarsiaml-golden'; DD='$HOME/intarsia-tp2/dumps'  # expanded on the Sparks
# rank 1 on spark2: a loop of tp-workers (one per rank-0 connection) until stop-worker. WENV: worker env; WWRAP: a
# command prefix (nsys); WBIN: another binary.
worker_start() { # <tag> <flags...>
  local tag=$1; shift
  r spark2 "rm -f ~/intarsia-tp2/stop-worker; nohup bash -c 'i=0; while [ ! -f ~/intarsia-tp2/stop-worker ]; do i=\$((i+1)); env $NCCLENV ${XENV:-} ${WENV:-} ${WWRAP:-} ${WBIN:-$BIN} tp-worker --listen $WIP:$WPORT $* >> $XO/worker-$tag.log 2>&1; echo \"worker exit \$? (\$i)\" >> $XO/worker-$tag.log; sleep 1; done' > /dev/null 2>&1 < /dev/null &"
  sleep 1
}
worker_stop() {
  r spark2 "touch ~/intarsia-tp2/stop-worker; pkill -TERM -f '[t]p-worker --listen'; for i in \$(seq 1 30); do pgrep -f '[t]p-worker --listen' >/dev/null || break; sleep 1; done; pkill -KILL -f '[t]p-worker --listen'; true"
}
# greedy ids: <tag> <chars> <binary> <extra flags>; ids to $XO/gen-<tag>.ids, worker's graph line logged
GEN_FLAGS="-m $M35 --backend cuda --tp 2 --max-seq 20480 --mem-budget 16 --profile-every 0"
xgen() {
  local tag=$1 chars=$2 bin=$3; shift 3
  WBIN=$bin worker_start $tag $GEN_FLAGS "$@"
  r spark1 "env $NCCLENV ${XENV:-} timeout ${GEN_TIMEOUT:-600} $bin generate $GEN_FLAGS $* --tp-remote $WIP:$WPORT --chat --no-think --ignore-eos --show-ids \
    -p \"\$(head -c $chars ~/intarsia-tp2/prompts/corpus_docs.txt)\" --max-tokens 256 > $XO/gen-$tag.out 2> $XO/gen-$tag.err; echo $tag rc=\$?; grep -aE 'finish=|graph' $XO/gen-$tag.err | head -3; grep -a '^ids:' $XO/gen-$tag.err > $XO/gen-$tag.ids" | tee -a "$L/steps.log"
  worker_stop
  rank1_check $tag "$@"
}
# rank 1's graph counters for run <tag>; a graphs run (no --no-graphs in <flags>) must replay: else REPLAY-CHECK FAIL
rank1_check() { # <tag> <flags...>
  local tag=$1; shift
  local line; line=$(r spark2 "grep -a 'decode graph' $XO/worker-$tag.log | tail -1")
  echo "  rank1 $tag: $line" | tee -a "$L/steps.log"
  if [[ " $* " != *" --no-graphs "* ]]; then
    [[ $line =~ replays=([0-9]+) ]] && (( ${BASH_REMATCH[1]} > 0 )) && log "  REPLAY-CHECK ok $tag" || log "  REPLAY-CHECK FAIL $tag: rank 1 did not replay"
  fi
}
# ids compare: <a> <b> (tags)
idcmp() {
  rsync -a "spark1:intarsia-tp2/xout/$OUT/" "$L/"
  python3 - "$L" "$1" "$2" <<'PY' | tee -a "$L/steps.log"
import sys, re, os
d, a, b = sys.argv[1:4]
def ids(t):
    f = f"{d}/gen-{t}.ids"
    if not os.path.exists(f) or not open(f).read().strip():
        return None
    return [int(v) for v in re.findall(r"-?\d+", open(f).read().split(":", 1)[1])]
x, y = ids(a), ids(b)
if x is None or y is None:
    print(f"IDS {a} vs {b}: missing ({'ok' if x else 'none'}/{'ok' if y else 'none'})")
else:
    k = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    same = x == y
    print(f"IDS {a} vs {b}: {'IDENTICAL' if same else 'DIFFER'} (first divergence {k if k is not None else 'none'}, {len(x)}/{len(y)})")
PY
}
AF="-m $M35 --backend cuda --tp 2 --max-seq 16384 --prefill-chunk 2048 --mem-budget 18"
xacc() { # <tag> <binary> <accuracy args...> (--no-graphs among them goes to both ranks)
  local tag=$1 bin=$2; shift 2; want $tag || return 0
  local wf=""; [[ " $* " == *" --no-graphs "* ]] && wf="--no-graphs"
  WBIN=$bin worker_start $tag $AF $wf
  r spark1 "env $NCCLENV ${XENV:-} timeout ${ACC_TIMEOUT:-1500} $bin accuracy $AF --tp-remote $WIP:$WPORT $* > $XO/$tag.log 2>&1; echo $tag rc=\$?; grep -aE 'SELF-STATS|self-golden gate|floor verdict|prefix-reuse gate|verdict|^Error' $XO/$tag.log | cut -c1-260" | tee -a "$L/steps.log"
  worker_stop
  rank1_check $tag $wf
}
B_SHORT="--self-max-kl 1.8e-2 --self-max-pos-kl 2.3 --self-max-flip-pct 3.76 --self-max-tail 2.8e-3 --self-prompt-factor 6.5"
# graphs vs eager on the same build: bitwise expected, so judge at the floor (any real change fails)
B_FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8 --self-prompt-factor 1000"
B_LONG="--self-max-kl 5.7e-2 --self-max-pos-kl 54.6 --self-max-flip-pct 8.7 --self-max-tail 7.1e-3 --self-prompt-factor 5.9"

# ---- gates (vLLM up) ----
if [[ $MODE == gates ]]; then
  r spark1 "mkdir -p ~/intarsia-tp2/dumps"
  # 1. tp=2 graphs vs tp=2 eager: 256 greedy ids at ~1.5k and ~16k tokens (bitwise expected: same kernels, 2-operand sums)
  for sz in 5k:5000 16k:54000; do
    t=${sz%%:*}; c=${sz#*:}
    want ids$t || continue
    xgen tp2e-$t $c $BIN --no-graphs
    xgen tp2g-$t $c $BIN
    idcmp tp2e-$t tp2g-$t
  done
  # 2. logits: tp=2 eager dumps (with the vLLM floor verdict), then tp=2 graphs judged against them; prefix reuse with graphs
  xacc eager-short $BIN --golden $VG/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --no-graphs --write-self-golden $DD/T2E-short.safetensors
  xacc graph-self-short $BIN --self-golden $DD/T2E-short.safetensors --mode both --allow-config-mismatch --pre-change ${PRE_CHANGE:?} --allow-same-code $B_FLOOR
  # the tp=2 floor: eager against its own dump (cross-host run-to-run), the reference for the graphs row
  xacc eager-self-short $BIN --self-golden $DD/T2E-short.safetensors --mode both --no-graphs --allow-config-mismatch --pre-change $PRE_CHANGE --allow-same-code $B_FLOOR
  xacc eager-long $BIN --golden $VG/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode both --no-graphs --write-self-golden $DD/T2E-long.safetensors
  xacc graph-self-long $BIN --self-golden $DD/T2E-long.safetensors --mode both --allow-config-mismatch --pre-change $PRE_CHANGE --allow-same-code $B_FLOOR
  xacc graph-floor-short $BIN --golden $VG/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both
  xacc graph-floor-long $BIN --golden $VG/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode both
  xacc graph-reuse $BIN --golden $VG/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode reuse
  # 2b. row-local fusion under TP (opt-in INTARSIA_TP_FUSION=1, both ranks): ids against tp=2 graphs unfused, the vLLM floor
  # (PR #40 judge) with dumps for the run-to-run KL table, the 35B TP self-golden budget against tp=2 eager, prefix reuse
  for sz in 5k:5000 16k:54000; do
    t=${sz%%:*}; c=${sz#*:}
    want fids$t || continue
    xgen tp2g-$t $c $BIN
    XENV=INTARSIA_TP_FUSION=1 xgen tp2gf-$t $c $BIN
    idcmp tp2g-$t tp2gf-$t
    idcmp tp2e-$t tp2gf-$t
  done
  XENV=INTARSIA_TP_FUSION=1 xacc fus-floor-short $BIN --golden $VG/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --write-self-golden $DD/T2F-short.safetensors
  XENV=INTARSIA_TP_FUSION=1 xacc fus-floor-long $BIN --golden $VG/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode both --write-self-golden $DD/T2F-long.safetensors
  XENV=INTARSIA_TP_FUSION=1 xacc fus-self-short $BIN --self-golden $DD/T2E-short.safetensors --mode both --allow-config-mismatch --pre-change ${PRE_EAGER:-$PRE_CHANGE} --allow-same-code $B_SHORT
  XENV=INTARSIA_TP_FUSION=1 xacc fus-self-long $BIN --self-golden $DD/T2E-long.safetensors --mode both --allow-config-mismatch --pre-change ${PRE_EAGER:-$PRE_CHANGE} --allow-same-code $B_LONG
  XENV=INTARSIA_TP_FUSION=1 xacc fus-reuse $BIN --golden $VG/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode reuse
  # 2c. which fused ops run under TP fusion: rank 0's op table, 1k prompt, 32 tokens, with and without INTARSIA_TP_FUSION
  if want fprobe; then
    for v in u f; do
      xe=""; [[ $v == f ]] && xe=INTARSIA_TP_FUSION=1
      XENV=$xe worker_start fprobe-$v $GEN_FLAGS
      r spark1 "env $NCCLENV $xe timeout 300 $BIN generate $GEN_FLAGS --tp-remote $WIP:$WPORT --op-table --chat --no-think --ignore-eos -p \"\$(head -c 3500 ~/intarsia-tp2/prompts/corpus_docs.txt)\" --max-tokens 32 > $XO/fprobe-$v.out 2> $XO/fprobe-$v.err; echo fprobe-$v rc=\$?; grep -aE '^(matmul_pair|matmul_fp8_pair|rmsnorm_add|gdn_conv_delta|gated_rmsnorm|qk_norm_rope_kv|gate_split_qk|matmul_nvfp4_glu|moe_block|matmul_fp8 |rmsnorm )' $XO/fprobe-$v.err | cut -c1-90" | tee -a "$L/steps.log"
      worker_stop
    done
  fi
  # 3. tp=1 unchanged: plan show (35B, 8B), 256 greedy ids, self-golden of the dev build against the base build (spark1 alone)
  if want tp1; then
    M8=$(r spark1 "ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/* 2>/dev/null | head -1")
    for b in $BBIN $BIN; do
      n=$([[ $b == "$BIN" ]] && echo dev || echo base)
      r spark1 "$b plan show -m $M35 --backend cuda --max-seq 20480 --mem-budget 29 --json > $XO/plan35-$n.json 2>/dev/null; [ -n '$M8' ] && $b plan show -m $M8 --backend cuda --mem-budget 29 --json > $XO/plan8-$n.json 2>/dev/null; true"
    done
    r spark1 "cmp $XO/plan35-base.json $XO/plan35-dev.json && echo 'plan 35B tp1 base vs dev: IDENTICAL' \$(wc -c < $XO/plan35-dev.json); cmp $XO/plan8-base.json $XO/plan8-dev.json && echo 'plan 8B tp1 base vs dev: IDENTICAL' \$(wc -c < $XO/plan8-dev.json)" | tee -a "$L/steps.log"
    T1F="-m $M35 --backend cuda --max-seq 20480 --mem-budget 29 --profile-every 0"
    for sz in 5k:5000 16k:54000; do
      t=${sz%%:*}; c=${sz#*:}
      for b in $BBIN $BIN; do
        n=$([[ $b == "$BIN" ]] && echo dev || echo base)
        r spark1 "timeout 600 $b generate $T1F --chat --no-think --ignore-eos --show-ids -p \"\$(head -c $c ~/intarsia-tp2/prompts/corpus_docs.txt)\" --max-tokens 256 > $XO/gen-tp1$n-$t.out 2> $XO/gen-tp1$n-$t.err; grep -a '^ids:' $XO/gen-tp1$n-$t.err > $XO/gen-tp1$n-$t.ids; grep -aE 'graph' $XO/gen-tp1$n-$t.err | head -1" | sed "s/^/  tp1 $n $t: /" | tee -a "$L/steps.log"
      done
      idcmp tp1base-$t tp1dev-$t
    done
    A1="-m $M35 --backend cuda --max-seq 16384 --prefill-chunk 2048 --mem-budget 29"
    r spark1 "timeout 1500 $BBIN accuracy $A1 --golden $VG/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --write-self-golden $DD/T1B-short.safetensors > $XO/tp1-base-short.log 2>&1; echo rc=\$?; \
      timeout 1500 $BIN accuracy $A1 --self-golden $DD/T1B-short.safetensors --mode both --allow-config-mismatch --pre-change ${PRE_BASE:?} $B_SHORT > $XO/tp1-self-short.log 2>&1; echo rc=\$?; grep -aE 'SELF-STATS|verdict' $XO/tp1-*-short.log | cut -c1-260" | tee -a "$L/steps.log"
  fi
  # 4. the kernel admission sweep (no kernel changed; the gate asks for it)
  if want kernels; then
    r spark1 "timeout 1500 $BIN kernels test --backend cuda > $XO/kernels-test.log 2>&1; echo kernels rc=\$?; grep -cE 'FAIL' $XO/kernels-test.log; tail -3 $XO/kernels-test.log" | tee -a "$L/steps.log"
  fi
fi

# ---- mutants (vLLM up): each against the eager ids of the gates run ($REF) and a self-golden run ----
if [[ $MODE == mut ]]; then
  REF=${REF:?gates OUT dir with gen-tp2e-5k.ids}
  cp "$RAW/$REF/gen-tp2e-5k.ids" "$L/gen-ref-5k.ids"
  for m in ${MUTANTS:-no-peer-upload no-rekey no-reserve peer-ignores-flag drop-captured-allreduce}; do
    MB="\$HOME/intarsia-tp2/bin/g1c-mut-$m-intarsia"
    log "== mutant $m"
    GEN_TIMEOUT=300 xgen mut-$m-5k 5000 $MB
    idcmp ref-5k mut-$m-5k
    ACC_TIMEOUT=900 xacc mut-$m-self $MB --self-golden $DD/T2E-short.safetensors --mode both --allow-config-mismatch --pre-change ${PRE_CHANGE:?} --allow-same-code --allow-dirty $B_FLOOR
  done
fi

# ---- window (vLLM down): nsys profile of both ranks, then interleaved A/B x3 of the serving arms ----
SFLAGS="-m $M35 --backend cuda --max-seq ${SMAXSEQ:-20480} --mem-budget ${MEMB:-45}"
bench_arm() { # <arm> <round> <rank-0 env> <worker flags or "-" for tp=1> <serve extra...>
  local arm=$1 rd=$2 senv=$3 wflags=$4; shift 4
  (( $(left) > 120 )) || { log "SKIP arm $arm round $rd: $(left) s left"; return 0; }
  local o=$XO/$arm-r$rd
  if [[ $wflags != - ]]; then WENV="$senv" worker_start $arm-r$rd --serving $SFLAGS --tp 2 $wflags; fi
  r spark1 "mkdir -p $o; nohup env $NCCLENV $senv $BIN serve $SFLAGS --port 8090 --host 0.0.0.0 --served-model-name intarsia-35b $* > $o/serve.log 2>&1 < /dev/null & echo \$! > $o/serve.pid"
  local up=0; for i in $(seq 1 150); do curl -sf -m 3 http://spark-4e77:8090/health >/dev/null && { up=1; break; }; sleep 2; done
  if [[ $up == 1 ]]; then
    local gx=""; [[ $arm == *g* ]] && gx="--require-graphs"
    r spark1 "$BIN bench speed --url http://127.0.0.1:8090 --model intarsia-35b --sizes ${SIZES:-1024,4096,16384} --gen 128 --reps ${REPS:-2} --warmup 1 --thinking off --at-most $gx --label g1c-$arm-r$rd --out-dir $o > $o/bench.log 2>&1; echo $arm r$rd rc=\$?; tail -n 5 $o/bench.log | cut -c1-170" | tee -a "$L/steps.log"
    if [[ ${LOCALEVAL:-} == "$arm" && $rd == 1 ]] && (( $(left) > 240 )); then
      log "localeval arm $arm"
      (cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url http://spark-4e77:8090/v1 --model intarsia-35b \
        --thinking off --sizes 1024,4096,16384 --gen 128 --reps 3 --label intarsia-35b-tp2-graphs-xhost) > "$L/localeval-$arm.log" 2>&1
      log "localeval arm $arm exit $?: $(tail -n 12 "$L/localeval-$arm.log" | tr '\n' '|' | cut -c1-600)"
    fi
  else
    log "arm $arm r$rd: server did not come up"; r spark1 "tail -5 $o/serve.log"
  fi
  r spark1 "kill \$(cat $o/serve.pid); for i in \$(seq 1 30); do kill -0 \$(cat $o/serve.pid) 2>/dev/null || break; sleep 1; done; kill -9 \$(cat $o/serve.pid) 2>/dev/null; true"
  if [[ $wflags != - ]]; then worker_stop; rank1_check $arm-r$rd $wflags; fi
}
if [[ $MODE == window || " ${STEPS:-} " == *" prof "* ]]; then
  if want prof; then
    # graphs, 1k prompt, 64 tokens; both ranks under nsys with per-node graph tracing (the NCCL kernel of every all-reduce)
    PF="-m $M35 --backend cuda --tp 2 --max-seq 8192 --mem-budget 16 --profile-every 0"
    NS="nsys profile --cuda-graph-trace=node -t cuda --sample=none --cpuctxsw=none -f true"
    r spark1 "timeout 300 $NS -o $XO/nsys-tp1-g $BIN generate -m $M35 --backend cuda --max-seq 8192 --mem-budget 29 --profile-every 0 --chat --no-think --ignore-eos -p \"\$(head -c 3500 ~/intarsia-tp2/prompts/corpus_docs.txt)\" --max-tokens 64 > $XO/prof-tp1.out 2> $XO/prof-tp1.err; echo prof-tp1 rc=\$?; grep -aE 'decode|graph' $XO/prof-tp1.err | head -2; cd ~/intarsia-tp2/xout/$OUT && nsys stats -q --report cuda_gpu_trace --format csv -o nsys-tp1-g nsys-tp1-g.nsys-rep >/dev/null 2>&1; ls nsys-tp1-g*" | tee -a "$L/steps.log"
    for v in ${PROF_VARIANTS:-g e}; do
      wf=""; [[ $v == e ]] && wf="--no-graphs"
      xe=""; [[ $v == gf ]] && xe=INTARSIA_TP_FUSION=1
      XENV=$xe WWRAP="$NS -o $XO/nsys-r1-$v" worker_start prof-$v $PF $wf
      # one worker only: a restarted nsys worker would overwrite the report before the first one is written
      r spark2 "sleep 2; touch ~/intarsia-tp2/stop-worker"
      r spark1 "env $NCCLENV $xe timeout 300 $NS -o $XO/nsys-r0-$v $BIN generate $PF $wf --tp-remote $WIP:$WPORT --chat --no-think --ignore-eos -p \"\$(head -c 3500 ~/intarsia-tp2/prompts/corpus_docs.txt)\" --max-tokens 64 > $XO/prof-$v.out 2> $XO/prof-$v.err; echo prof-$v rc=\$?; grep -aE 'decode|graph' $XO/prof-$v.err | head -3" | tee -a "$L/steps.log"
      r spark2 "for i in \$(seq 1 60); do pgrep -f '[t]p-worker --listen' >/dev/null || break; sleep 1; done"; worker_stop; sleep 3
      for h in spark1 spark2; do
        rk=$([[ $h == spark1 ]] && echo 0 || echo 1)
        r $h "cd ~/intarsia-tp2/xout/$OUT && ls nsys-r$rk-$v.nsys-rep >/dev/null 2>&1 && nsys stats -q --report cuda_gpu_trace --format csv -o nsys-r$rk-$v nsys-r$rk-$v.nsys-rep >/dev/null 2>&1; ls -la nsys-r$rk-$v* 2>&1 | tail -3" | tee -a "$L/steps.log"
      done
    done
  fi
fi
if [[ $MODE == window ]]; then
  # A/B: tp1g = tp=1 graphs (the shipped config), tp2e = tp=2 eager (PR #44), tp2g = tp=2 graphs; order reversed in even rounds
  ARMS_FWD=${ARMS:-tp1g tp2g}
  for rd in $(seq 1 ${ROUNDS:-3}); do
    order=$ARMS_FWD; (( rd % 2 == 0 )) && order=$(echo $ARMS_FWD | tr ' ' '\n' | tail -r | tr '\n' ' ')
    (( rd % 2 == 1 )) && order="$order ${EXTRA_ARMS:-tp2e}"
    for a in $order; do
      case $a in
        tp1g) bench_arm tp1g $rd "" - ;;
        tp2e) bench_arm tp2e $rd "" "--no-graphs" --tp 2 --tp-remote $WIP:$WPORT --no-graphs ;;
        tp2g) bench_arm tp2g $rd "" "" --tp 2 --tp-remote $WIP:$WPORT ;;
        tp2gf) bench_arm tp2gf $rd "INTARSIA_TP_FUSION=1" "" --tp 2 --tp-remote $WIP:$WPORT ;;
        tp2g-ll) SIZES=1024 bench_arm tp2g-ll $rd "NCCL_PROTO=LL" "" --tp 2 --tp-remote $WIP:$WPORT ;;
        tp2g-ll128) SIZES=1024 bench_arm tp2g-ll128 $rd "NCCL_PROTO=LL128" "" --tp 2 --tp-remote $WIP:$WPORT ;;
        tp2g-simple) SIZES=1024 bench_arm tp2g-simple $rd "NCCL_PROTO=Simple" "" --tp 2 --tp-remote $WIP:$WPORT ;;
        *) log "unknown arm $a" ;;
      esac
    done
  done
fi
log "== $MODE done, $(left) s of the work budget left"
