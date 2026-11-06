#!/usr/bin/env bash
# Lane bm: Metal gates on endor, ON endor inside one fleetq slot (--exclusive):
#   fleetq run --host endor --exclusive --mem 40 --expect 150 --label "bm: Metal no-op + batched MTP ids + speed" -- scripts/bm/metal-noop.sh
# (1) no-op vs main 602a1fcfe: plan show (text + json) and 256 greedy ids, Qwen3-8B and the 35B NVFP4, short and ~3.5k prompts
#     (scripts/tp2-hip/noop/noop.sh); kernels test; goldens qwen3-8b-fp32 / -long (bf16 gate class) on the branch.
# (2) batched MTP greedy ids on the Metal 35B: solo spec-off server vs batched-MTP servers (phase A and B), 6 concurrent prompts.
# (3) localeval ABBA, 2 rounds: plain batched vs phase A (B<=1) vs phase B (B<=5), 1/2/4/8 streams.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
DEV="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
S=$DEV/.scratch/metal; O=$DEV/docs/results/2026-11-05-batched-mtp/raw/metal; mkdir -p $S $O
M8=$HOME/models/Qwen3-8B; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
if [ ! -d $S/base-src ]; then mkdir -p $S/base-src && git -C $DEV archive 602a1fcfe | tar -x -C $S/base-src; fi
for t in base dev; do
  if [ $t = base ]; then T=$S/base-src; else T=$DEV; fi
  echo "== build $t $(date +%T)"
  (cd $T && CARGO_TARGET_DIR=$S/target-$t cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  RUN="$S/target-$t/release/intarsia" bash $DEV/scripts/tp2-hip/noop/noop.sh run $O $t metal $M8 $M35 $DEV/scripts/tp2-hip/noop/prompts
done
bash $DEV/scripts/tp2-hip/noop/noop.sh cmp $O base dev | tee $O/noop.txt
B=$S/target-dev/release/intarsia
echo "== kernels test metal $(date +%T)"
$B kernels test --backend metal < /dev/null > $O/kernels.log 2>&1; tail -3 $O/kernels.log
echo "== goldens $(date +%T)"
for g in qwen3-8b-fp32 qwen3-8b-long-fp32; do
  $B accuracy --backend metal --model $M8 --golden /opt/code/intarsiaml/golden/$g.safetensors --mode both --gate-class bf16 < /dev/null > $O/golden-$g.log 2>&1
  echo "$g exit $?"; tail -4 $O/golden-$g.log
done
PORT=8193; SPID=""
stop() { [[ -n "$SPID" ]] && { kill $SPID 2>/dev/null; for i in $(seq 1 20); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; SPID=""; }; }
trap stop EXIT
serve() { # serve <log> <env> <args...>
  local log=$1 envs=$2; shift 2
  env $envs $B serve --backend metal --model $M35 --port $PORT --host 127.0.0.1 --served-model-name m --max-seq 8192 --profile-every 0 "$@" > $log 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 150); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && return 0; sleep 2; done
  echo "server did not come up: $log"; tail -5 $log; stop; return 1
}
ids() { # ids <name> <env> <conc> <args...>
  local name=$1 envs=$2 conc=$3; shift 3
  echo "== ids $name $(date +%T)"
  serve $O/serve-ids-$name.log "$envs" "$@" && python3 $DEV/scripts/bm/ids.py http://127.0.0.1:$PORT/v1 m $O/ids-$name.json $conc
  curl -sf http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_spec_(proposed|accepted)_tokens_total' | tee $O/ids-$name.metrics
  stop
}
ids solo-off "" 1 --max-batch 1 --spec off
ids bmtp4 "INTARSIA_BATCHED_MTP=4" 6 --max-batch 4 --spec mtp --spec-k 2 --spec-off-at 5
ids bver5 "INTARSIA_BATCHED_MTP=5 INTARSIA_BATCHED_VERIFY=1" 6 --max-batch 8 --spec mtp --spec-k 2 --spec-off-at 9
ids plain8 "" 6 --max-batch 8
for n in bmtp4 bver5 plain8; do
  python3 - "$O/ids-solo-off.json" "$O/ids-$n.json" <<'PY' | tee -a $O/ids.txt
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = [x == y for x, y in zip(a, b)]
print(sys.argv[2].split('/')[-1], "identical per sequence:", same, "ALL" if all(same) and len(a) == len(b) else "MISMATCH")
PY
done
echo "== speed $(date +%T)"
le() { # le <label> <recipe>
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model m --thinking off --sizes 1024 --gen 256 --reps 3 \
      --concurrency 1,2,4,8 --label "$1" --no-unim --recipe "$2" ) > $O/$1.out 2>&1
  echo "$1: $(grep -aE 'run_id|Run id|results/' $O/$1.out | head -2 | tr '\n' ' ')" | tee -a $O/index.txt
  grep -aE 'streams: aggregate' $O/$1.out
}
ARMS=("plain||--max-batch 8" "bmtp1|INTARSIA_BATCHED_MTP=1|--max-batch 8 --spec mtp --spec-k 2" "bver5|INTARSIA_BATCHED_MTP=5 INTARSIA_BATCHED_VERIFY=1|--max-batch 8 --spec mtp --spec-k 2 --spec-off-at 9")
run_arm() {
  local r=$1 name envs args; IFS='|' read -r name envs args <<< "$2"
  echo "== round $r arm $name $(date +%T)"
  serve $O/serve-$name-r$r.log "$envs" $args || return
  le "bm-metal-$name-r$r" "intarsia serve --backend metal --max-seq 8192 env='$envs' $args"
  curl -sf http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_spec_(proposed|accepted)_tokens_total' > $O/bm-metal-$name-r$r.metrics
  stop
}
for r in 1 2; do
  if (( r % 2 == 0 )); then for ((k=${#ARMS[@]}-1; k>=0; k--)); do run_arm $r "${ARMS[k]}"; done
  else for a in "${ARMS[@]}"; do run_arm $r "$a"; done; fi
done
echo "== done $(date +%T)"
