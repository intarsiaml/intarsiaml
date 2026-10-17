#!/bin/zsh
# GPU part of the fatal-failure test (M2 item D3): the end-to-end fault injection through `intarsia serve`, both modes, plus the
# Metal backend's own tests. Takes no fleetq slot itself: run it as the command of ONE exclusive slot (it uses the GPU):
#
#   fleetq run --host endor --exclusive --mem 24 --label "cb fault-inject e2e" -- tools/cbfault_slot.sh OUTDIR
#
# tools/cbfault_slot.sh OUTDIR
# env: BIN (a `--features metal,fault-inject` release binary; default target/release/intarsia of this worktree),
#      MODEL (~/models/Qwen3-8B), PORT (8117), MODES (err,panic), CASES (nosup,sup), SKIP_CARGO=1 (skip the cargo tests).
# Build first, outside the slot, so the slot holds the GPU only for the runs:
#   cargo build --release -p intarsia-cli --features metal,fault-inject
#   cargo test -p intarsia-backend-metal --release --no-run
#   cargo test -p intarsia-backend-metal --release --features fault-inject --test fault_inject --no-run
# Only processes the driver started are ever signalled (its own process group); it checks nothing is left behind.
set -u
OUT=${1:?usage: cbfault_slot.sh OUTDIR}; mkdir -p "$OUT"; OUT=${OUT:A}
T=${0:A:h}; ROOT=${T:h}
BIN=${BIN:-$ROOT/target/release/intarsia}
MODEL=${MODEL:-$HOME/models/Qwen3-8B}
PORT=${PORT:-8117}
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -x "$BIN" ] || { echo "no binary $BIN" >&2; exit 3; }
[ -n "${FLEETQ_JOB:-}" ] || echo "WARNING: FLEETQ_JOB not set: not inside a fleetq slot" >&2
exec > >(tee -a "$OUT/slot.log") 2>&1
echo "== $(date '+%F %T') cbfault_slot: bin=$BIN ($(stat -f %z "$BIN") bytes) model=$MODEL port=$PORT"
{ echo "--- hygiene start $(date '+%H:%M:%S')"; ps -axo pid,pcpu,rss,etime,comm | sort -k2 -nr | head -8 | cut -c1-120; sysctl vm.swapusage; } > "$OUT/hygiene.txt" 2>&1
RC=0
echo "== e2e"
python3 "$T/cbfault_e2e.py" --bin "$BIN" --model "$MODEL" --out "$OUT/e2e" --modes "${MODES:-err,panic}" --cases "${CASES:-nosup,sup}" --port $PORT
R=$?; echo "cbfault_e2e rc=$R"; [ $R -gt $RC ] && RC=$R
if [ -z "${SKIP_CARGO:-}" ]; then
  cd "$ROOT" || exit 3
  echo "== cargo test -p intarsia-backend-metal --release"
  cargo test -p intarsia-backend-metal --release > "$OUT/cargo-metal.log" 2>&1; R=$?
  echo "cargo test metal rc=$R: $(grep -E '^test result' "$OUT/cargo-metal.log" | awk '{p+=$4; f+=$6} END {print p " passed, " f " failed"}')"; [ $R -gt $RC ] && RC=$R
  echo "== cargo test -p intarsia-backend-metal --release --features fault-inject --test fault_inject"
  cargo test -p intarsia-backend-metal --release --features fault-inject --test fault_inject > "$OUT/cargo-metal-fault.log" 2>&1; R=$?
  echo "cargo test metal fault_inject rc=$R: $(grep -E '^test result' "$OUT/cargo-metal-fault.log")"; [ $R -gt $RC ] && RC=$R
fi
echo "== leftover check"
pgrep -fl "intarsia.*serve.*--port $PORT" && { echo "LEFTOVER serve process"; RC=9; } || echo "no leftover serve process"
lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null | grep . && { echo "port $PORT still listening"; RC=9; } || echo "port $PORT free"
{ echo "--- hygiene end $(date '+%H:%M:%S')"; ps -axo pid,pcpu,rss,etime,comm | sort -k2 -nr | head -8 | cut -c1-120; } >> "$OUT/hygiene.txt" 2>&1
echo "== done rc=$RC"
exit $RC
