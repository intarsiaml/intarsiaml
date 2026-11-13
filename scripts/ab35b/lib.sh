#!/usr/bin/env bash
# ab35b shared helpers (sourced). Everything runs inside a fleetq slot (FLEETQ_JOB set), so localeval does not queue again.
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-12-ab35b-cutover-followup/raw; mkdir -p "$OUT"
S=$REPO/scripts/ab35b
HN=spark-fa00
RDIR=$REPO/.ab35b/recipes; mkdir -p "$RDIR"
# recipe_file <name> <text>: localeval stats a --recipe string as a path, so a long literal fails ("File name too long"): pass a file
recipe_file() { printf '%s\n' "$2" > "$RDIR/$1.txt"; echo "$RDIR/$1.txt"; }
keep() { # keep <out-log> <label>: copy the localeval run's summary/result next to the log and note the run id
  local rid; rid=$(grep -aoE 'results/[0-9]{8}-[^ ]*' "$1" | head -1 | sed 's#results/##; s#/.*##')
  echo "$2 run_id=$rid" | tee -a "$OUT/index.txt"
  [[ -n $rid && -d /opt/code/deepeval/results/$rid ]] && { mkdir -p "$OUT/$rid"; cp /opt/code/deepeval/results/$rid/summary.md /opt/code/deepeval/results/$rid/result.json "$OUT/$rid/" 2>/dev/null; }
}
# gsm8k <label> <url> <recipe-file>: fixed prompt (pinned template, NO nonce), n=200, thinking off, temperature 0 (localeval sends it)
gsm8k() {
  echo "== $1 gsm8k $(date +%T)"
  ( cd /opt/code/deepeval && uv run localeval quality --url "$2" --model guidance --thinking off --structured prompt --wait-idle 600 --recipe "$3" \
      --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k --label "ab35b-$1-gsm8k" ) > "$OUT/$1-gsm8k.out" 2>&1
  tail -3 "$OUT/$1-gsm8k.out"; keep "$OUT/$1-gsm8k.out" "$1 gsm8k"
}
# replay <arm> <url> <order> <seed> : multi-turn replay, plan/trace shared by seed
replay() {
  echo "== $1 replay $3 seed $4 $(date +%T)"
  ( cd /opt/code/deepeval && uv run python "$S/replay.py" --url "$2" --model guidance --arm "$1" --order "$3" --seed "$4" \
      --plan "$OUT/replay-plan.json" --trace "$OUT/replay-trace-seed$4.json" --out "$OUT" ) > "$OUT/$1-replay-$3-seed$4.out" 2>&1
  tail -12 "$OUT/$1-replay-$3-seed$4.out"
}
sampled() { # sampled <arm> <url> [metrics-url] [sizes]
  echo "== $1 sampled $(date +%T)"
  ( cd /opt/code/deepeval && uv run python "$S/sampled.py" --url "$2" --model guidance --arm "$1" --out "$OUT" ${3:+--metrics "$3"} ${4:+--sizes "$4"} ) > "$OUT/$1-sampled.out" 2>&1
  tail -6 "$OUT/$1-sampled.out"
}
tools() { # tools <arm> <url>
  echo "== $1 tools $(date +%T)"
  python3 "$S/toolsmoke.py" --url "$2" --model guidance --arm "$1" --out "$OUT" > "$OUT/$1-tools.out" 2>&1
  tail -6 "$OUT/$1-tools.out"
}
