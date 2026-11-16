#!/bin/zsh
# Lane msmallm task 1: which M the 27B issues to the MXFP4 matmuls (GEMV vs NAX) per serving path. Run under fleetq --quiet gpu,cpu.
# INTARSIA_METAL_PROFILE=cb + COMMIT_EVERY=1 (one record per dispatch, rows in "t"), histogram by scripts/msmallm/mhist.py.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/msmallm; BIN=${BIN:-$W/target/release/intarsia}; R=$W/docs/results/2026-11-16-msmallm/raw/window; mkdir -p $R
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
SHORT="The history of the printing press begins"
P2K="$(head -c 9000 $PR/prompt-4096.txt)"   # about 2.1k tokens: one 2048-row chunk plus a short tail
PORT=18671
gen() { # tag spec prompt
  local SA=(); [ $2 = off ] && SA=(--spec off)
  env INTARSIA_METAL_PROFILE=cb:$R/$1.jsonl INTARSIA_METAL_COMMIT_EVERY=1 $BIN generate --model $M27 --backend metal --max-seq 40000 \
    --profile-every 0 --temperature 0 --max-tokens 8 --ignore-eos $SA --prompt "$3" > $R/$1.out 2> $R/$1.err
  echo "$1 exit $? $(grep -aoE 'prefill [0-9]+ tok.*' $R/$1.err | head -1)"
  python3 $W/scripts/msmallm/mhist.py $R/$1.jsonl; gzip -f $R/$1.jsonl
}
gen short-mtp default "$SHORT"
gen short-off off "$SHORT"
gen p2k-off off "$P2K"
gen p4k-mtp default "$(cat $PR/prompt-4096.txt)"
# Metal's batch cap
$BIN serve --backend metal --model $M27 --max-batch 16 --spec off --no-supervisor --no-ui --port $PORT > $R/mb16.out 2> $R/mb16.err < /dev/null &
P=$!; for i in {1..90}; do kill -0 $P 2>/dev/null || break; curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
kill -0 $P 2>/dev/null && { echo "max-batch 16: server UP"; kill -INT $P; wait $P; } || echo "max-batch 16: refused: $(grep -a -iE 'error|batch' $R/mb16.err | tail -2)"
# served: 9 concurrent short requests on --max-batch 8 --spec off, and 4 on the default engine (MTP, max-batch 1)
served() { # tag nreq extra...
  local tag=$1 n=$2; shift 2
  env INTARSIA_METAL_PROFILE=cb:$R/$tag.jsonl INTARSIA_METAL_COMMIT_EVERY=1 $BIN serve --backend metal --model $M27 --served-model-name intarsia \
    --max-seq 40000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@" > $R/$tag.out 2> $R/$tag.err < /dev/null &
  local P=$!; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  local C=()
  for j in $(seq 1 $n); do
    curl -s -m 200 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
      -d "{\"model\":\"intarsia\",\"messages\":[{\"role\":\"user\",\"content\":\"Request $j: write about the history of printing.\"}],\"max_tokens\":24,\"temperature\":0}" > $R/$tag.r$j.json &
    C+=($!)
  done
  for c in $C; do wait $c; done
  kill -INT $P; wait $P
  echo "$tag ($n requests $*):"; python3 $W/scripts/msmallm/mhist.py $R/$tag.jsonl; gzip -f $R/$tag.jsonl
}
served mb8-9req-off 9 --max-batch 8 --spec off
served mb1-4req-mtp 4
echo "## done"
