#!/bin/zsh
# One engine arm: start server, localeval quality (mmlu,gsm8k n=25, fx3 settings), stop server.
# Run inside a fleetq slot on endor: fleetq run --host endor --mem N -- zsh arm.sh <arm> <out dir>
# arms: intarsia8 mlx8 llama8 intarsia35
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
ARM=${1:?arm}; OUT=${2:?out}; mkdir -p $OUT; OUT=${OUT:A}
BIN=${BIN:?bin}
export INTARSIA_CPU_FREE_MEM_GB=${INTARSIA_CPU_FREE_MEM_GB:-60}
M8=/Users/josef/models/Qwen3-8B; G8=/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf
M35=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4
case $ARM in
 intarsia8) PORT=8093; HP=/health; MID=intarsia-metal; CMD=($BIN serve --model $M8 --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name $MID --max-seq 8192 --profile-every 0 --no-supervisor --no-ui);;
 intarsia35) PORT=8093; HP=/health; MID=intarsia-metal; CMD=($BIN serve --model $M35 --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name $MID --max-seq 8192 --profile-every 0 --no-supervisor --no-ui);;
 mlx8) PORT=8092; HP=/v1/models; MID=$M8; CMD=(/Users/josef/.local/bin/mlx_lm.server --model $M8 --port $PORT --chat-template-args '{"enable_thinking":false}' --prompt-cache-size 1);;
 llama8) PORT=8091; HP=/health; MID=$G8; CMD=(/opt/homebrew/bin/llama-server -m $G8 -ngl 999 -fa on -c 8192 --cache-ram 0 --jinja --port $PORT);;
esac
$BIN --version > $OUT/$ARM-versions.txt 2>&1; echo "$CMD" >> $OUT/$ARM-versions.txt
$CMD > $OUT/$ARM-server.out 2> $OUT/$ARM-server.err < /dev/null &
SPID=$!
trap 'kill $SPID 2>/dev/null; sleep 2; kill -9 $SPID 2>/dev/null' EXIT
up=0; for i in {1..300}; do curl -sf http://127.0.0.1:$PORT$HP >/dev/null 2>&1 && { up=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 2; done
[ $up = 1 ] || { echo "SERVER FAILED" >&2; exit 1; }
cd /opt/code/deepeval
LE=(uv run localeval quality --url http://127.0.0.1:$PORT/v1 --model $MID --thinking off --structured none --bench mmlu --limit 25 --label mpre-$ARM --recipe "$CMD")
[ -n "${PROBE:-}" ] && { uv run localeval probe --url http://127.0.0.1:$PORT/v1 --model $MID --thinking off; exit $?; }
$LE > $OUT/$ARM-quality.out 2> $OUT/$ARM-quality.err
echo "localeval exit $?"
res=$(grep -o 'saved .*result.json' $OUT/$ARM-quality.out | tail -1 | sed 's/^saved //'); [ -n "$res" ] && cp $res $OUT/$ARM-result.json
