#!/usr/bin/env bash
# Lane cmtp: served A/B on spark1, arms differ by env (same binary), interleaved rounds. Run under fleetq run --quiet gpu,cpu:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "cmtp2: A/B s1" -- ssh spark1 'W=ab1 ARMS="base:PINNED_DOWNLOAD_OFF=1 (reverted experiment knob) dev:X=1" bash -s s1' < scripts/cmtp/ab.sh
# MODE s1: serve --max-batch 1 (MTP default), cells dec1k dec4k dec16k pf4k.  MODE s16: serve --max-batch 16, cells b1 b4 b8 b16 (1k prompts).
# ARMS "name:ENV=val[,ENV2=val]" (use X=1 for none); CELLS overrides the cell list; ROUNDS (3); REPS per cell (3, first dropped for dec).
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cmtp; W=${W:?W}; O=$R/out/$W; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/target-dev/release/intarsia}; PORT=${PORT:-8097}; ROUNDS=${ROUNDS:-3}; REPS=${REPS:-3}
MODE=${1:?s1|s16}; P=$R/src/scripts/dfh/ids-prompts; G=$R/src/scripts/cmtp/gen.py
ARMS=${ARMS:?ARMS}
case $MODE in
  s1) SARGS="--max-batch 1 --max-seq 20480"; CELLS=${CELLS:-"dec1k dec4k dec16k pf4k"} ;;
  s16) SARGS="--max-batch 16 --max-seq 4096"; CELLS=${CELLS:-"b1 b4 b8 b16"} ;;
esac
stop() { pkill -f "intarsia serve.*--port $PORT" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "intarsia serve.*--port $PORT" >/dev/null || break; sleep 1; done; }
trap stop EXIT
cell() { # cell <name> <arm> <round>
  local c=$1 a=$2 r=$3 out=$O/$2-r$3-$1.json
  case $c in
    dec1k)  python3 $G $PORT 1 384 $P/4k-0.txt 3700 $REPS d1k$r$a > $out ;;
    dec4k)  python3 $G $PORT 1 384 $P/4k-1.txt - $REPS d4k$r$a > $out ;;
    dec16k) python3 $G $PORT 1 256 $P/16k-0.txt - 2 d16k$r$a > $out ;;
    pf4k)   python3 $G $PORT 1 2 $P/4k-2.txt - 4 pf4k$r$a > $out ;;
    b1|b4|b8|b16) python3 $G $PORT ${c#b} 256 $P/4k-3.txt 3700 2 $c$r$a > $out ;;
  esac
  echo "$a r$r $c: $(tail -1 $out | cut -c1-200)"
}
for r in $(seq 1 $ROUNDS); do
  order="$ARMS"; [ $((r % 2)) = 0 ] && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for spec in $order; do
    a=${spec%%:*}; e=${spec#*:}
    stop
    env ${e//,/ } nohup $BIN serve --backend cuda --model $M35 --model-id m $SARGS --profile-every 0 --mem-budget 40 --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.log 2>&1 < /dev/null &
    for i in $(seq 1 100); do pgrep -f "intarsia serve.*--port $PORT" >/dev/null || { echo "serve died ($a)"; tail -5 $O/serve-$a-r$r.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
    python3 $G $PORT 1 48 $P/4k-0.txt 3700 2 warm$r$a > /dev/null 2>&1
    [ $MODE = s16 ] && python3 $G $PORT 16 32 $P/4k-0.txt 3700 1 warm$r$a > /dev/null 2>&1
    for c in $CELLS; do cell $c $a $r; done
    curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_spec_(steps|proposed|accepted)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt
    echo "$a r$r metrics: $(cat $O/metrics-$a-r$r.txt)"
  done
done
echo "ab done"
