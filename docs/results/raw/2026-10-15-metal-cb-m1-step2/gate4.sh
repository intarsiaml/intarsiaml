#!/bin/zsh
# cb-m1 step 2 gate 4 (throughput), one exclusive endor slot: usage gate4.sh <outdir> <rev>   (env GEN default 256, ORDER default "1 8 1 8 1 8")
# localeval 0.3.0 (feat/decode-aggregate): speed at ctx 1k, --gen $GEN, N = 1 (the sweep) and 2,4,8 streams, server --profile-every 0.
# --max-batch 1 (LocalEngine) and --max-batch 8 passes are INTERLEAVED in the same slot (ORDER) so drift shows up in the baseline;
# macmon logs the GPU clock once a second for the whole run; `ps` of other GPU-ish processes at the start and the end.
# Inside the slot FLEETQ_JOB is inherited by localeval.
set -u
setopt shwordsplit
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; REV=$2; mkdir -p $O
HERE=${0:A:h}
source $HERE/serve_lib.sh
snap() { echo "== $1 $(date +%T)" >> $O/ps-$2.txt; ps -axo pid,rss,%cpu,etime,command | grep -iE "kev|tensorfold|omlx|splash|llama|vllm|mlx|ollama|intarsia|localeval|python" | grep -v "grep\|ps -axo" | cut -c1-170 >> $O/ps-$2.txt; }
snap start start
macmon pipe -i 1000 > $O/macmon.jsonl 2> $O/macmon.err &
MPID=$!
trap 'kill $MPID 2>/dev/null; stop_server' EXIT
n=0
for MB in ${ORDER:-1 8 1 8 1 8}; do
  n=$((n+1))
  start_server p${n}-mb$MB 8201 --max-batch $MB || exit 3
  snap "pass $n mb$MB" pass
  (cd /opt/code/deepeval-decode-agg && uv run localeval speed --url http://127.0.0.1:8201/v1 --model Qwen3-8B --sizes 1024 --gen ${GEN:-256} --reps 3 --concurrency 2,4,8 --label cbwire-p$n-mb$MB-gen${GEN:-256}) > $O/localeval-p$n-mb$MB.log 2>&1; echo "pass $n localeval mb$MB: exit $?"
  curl -s http://127.0.0.1:8201/metrics > $O/metrics-p$n-mb$MB.txt
  stop_server
done
snap end end
