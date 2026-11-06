#!/usr/bin/env bash
# Served decode speed of sampled speculation (docs/results/2026-11-05-sampled-spec.md, rule 3), inside one exclusive endor slot.
# speed.sh <binary> <model dir> <tag> <out dir> <arms...>; arm = T<temp>-<off|mtp|match|reject>, temp in 00 07 10.
# One `intarsia serve --max-batch 1` per arm; `localeval suite --tier adhoc --cells speed-spec` (natural-v1 1k and 4k, 256 forced
# tokens) with a suite copy whose speed-spec sizes are [1024, 4096] (and lane plain for the spec-off arms, which the stock cell
# skips); the temperature comes from the target's extra_body (.scratch/suite/targets.toml). /metrics spec counters around each arm.
set -uo pipefail
bin=$1; model=$2; tag=$3; out=$4; shift 4
W=$(cd "$(dirname "$0")/../.." && pwd)
S=$W/.scratch/suite
port=18431
mkdir -p "$out"
for arm in "$@"; do
  t=${arm%%-*}; mode=${arm#*-}
  spec=(--spec mtp --spec-k 2); env_s=0; lane=spec:mtp; suite=$S/suite-spec.json
  case $mode in
    off) spec=(--spec off); lane=plain; suite=$S/suite-plain.json;;
    mtp) env_s=0;;
    match) env_s=match;;
    reject) env_s=reject;;
  esac
  cmd="INTARSIA_SPEC_SAMPLED=$env_s $bin serve --backend metal --model $model --port $port --model-id ss --max-seq 8192 --max-batch 1 --no-supervisor --no-ui ${spec[*]}"
  echo "== $tag $arm: $cmd"
  INTARSIA_SPEC_SAMPLED=$env_s $bin serve --backend metal --model "$model" --port $port --model-id ss --max-seq 8192 --max-batch 1 \
    --no-supervisor --no-ui "${spec[@]}" > "$out/$tag-$arm.serve.log" 2>&1 &
  pid=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:$port/health > /dev/null && break; sleep 1; done
  curl -s http://127.0.0.1:$port/metrics | grep -E "^intarsia_spec_(proposed|accepted)_tokens_total|^intarsia_spec_steps" > "$out/$tag-$arm.metrics0"
  ( cd /opt/code/deepeval && LOCALEVAL_TARGETS=$S/targets.toml uv run localeval suite ss-t${t#T} --tier adhoc --cells speed-spec \
      --lane $lane --suite "$suite" --tokenizer-file "$model/tokenizer.json" --label "ss-$tag-$arm" --recipe "$cmd" ) \
      > "$out/$tag-$arm.localeval.log" 2>&1
  echo "localeval exit $?"
  curl -s http://127.0.0.1:$port/metrics | grep -E "^intarsia_spec_(proposed|accepted)_tokens_total|^intarsia_spec_steps" > "$out/$tag-$arm.metrics1"
  kill $pid; wait $pid 2>/dev/null
  grep -E "decode|tok/s" "$out/$tag-$arm.localeval.log" | tail -6
done
