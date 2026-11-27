# sourced by the slot scripts of lane ink-metal-sampled2 (endor, under fleetq). Sets W S M G and helpers.
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-metal-sampled2; S=$W/.scratch
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
BIN=${BIN:-$S/bin/intarsia-b1}
IDLE=${IDLE:-45}
G="generate -m $M --backend metal --max-seq 8192 --profile-every 0 --show-ids --max-tokens 128 --ignore-eos --prompt-ids $S/prompts/speed.json --top-k 0 --top-p 1.0"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
stamp() { python3 -c 'import time; print(f"{time.time():.3f}")'; }
# guard: another intarsia generate / serve on the box (match the binary name at the start of the command, not our own shell)
guard() { local o; o=$(pgrep -fl '^[^ ]*/intarsia[^ ]* (serve|generate)' | head -3); [ -n "$o" ] && { echo "GUARD: another intarsia run on the box: $o"; exit 1; }; }
# run N seconds at most (perl alarm survives exec)
dl() { local n=$1; shift; perl -e 'alarm shift; exec @ARGV' $n "$@"; }
