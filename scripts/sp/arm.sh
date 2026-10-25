#!/usr/bin/env bash
# Lane sp, on endor, inside a fleetq slot that holds tatooine's cards with radiance down (FLEETQ_JOB=<id> reaches localeval).
#   arm.sh <tree: sp|sp-base> <tag> <plain|spec> "<serve args>" "<KEY=VAL ...>" [MAX_BATCH]
# Starts `intarsia serve` from the tree on tatooine (sp-win.sh up), runs the localeval suite cells for the arm, saves the
# server log (sp-win.sh down). plain: speed-plain + conc-plain-1k (filler text); spec: speed-spec (natural-v1 text).
set -uo pipefail
tree=$1; tag=$2; mode=$3; serve=$4; envs=${5:-}; mb=${6:-1}
TOK=${TOK:-/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/tok35}
LOG=${LOG:-/tmp/sp-arm}; mkdir -p "$LOG"
echo "[arm $tag] up $(date +%T)"
ssh tatooine "SERVE_ARGS='$serve' MAX_BATCH=$mb bash ~/sp-win.sh up $tree $tag $envs" 2>&1 | tail -3
if [ "$mode" = plain ]; then
  cells=speed-plain,conc-plain-1k; lane="--lane plain"
else
  cells=speed-spec; lane="--lane spec:mtp --drafter mtp"
fi
cd /opt/code/deepeval
timeout 1500 uv run localeval suite --url http://tatooine:8294/v1 --model intarsia --tier adhoc --cells "$cells" $lane \
  --tokenizer-file "$TOK" --max-context 20480 --max-num-seqs "$mb" --kv-dtype bf16 --label "sp-$tag" \
  --recipe "intarsia serve --backend hip --model Qwen3.6-35B-A3B-NVFP4 --max-seq 20480 --max-batch $mb $serve [$envs] tree=$tree" \
  > "$LOG/$tag.out" 2>&1
echo "[arm $tag] localeval rc=$? $(date +%T)"; tail -4 "$LOG/$tag.out" | cut -c1-200
ssh tatooine "bash ~/sp-win.sh snap $HOME/sp-logs/$tag.snap >/dev/null 2>&1; bash ~/sp-win.sh down $tag" 2>&1 | tail -2
