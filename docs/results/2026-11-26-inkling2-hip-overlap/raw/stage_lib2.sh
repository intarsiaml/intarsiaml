# Lane ink-hip-overlap2 additions to stage_lib.sh (sourced after it, ON tatooine): card 1's own clients pinned away from card 0's
# real-prompt client (BW_CPUS, default 4,5,16,17), and the Qwen3.6-35B-A3B-NVFP4 no-op helpers.
BW_CPUS=${BW_CPUS:-4,5,16,17}
SENV=${SENV:-}  # stage_lib.sh reads it under set -u (W0: an unset SENV stopped the first srv call)
bw() {  # name mode...: serve_bitwise.py against the server srv started (IP), pinned to BW_CPUS
  local name=$1 m ms; shift
  for m in "$@"; do
    (( $(left) < 15 )) && { echo "[$name] bitwise $m SKIP ($(left) s left)"; continue; }
    ms=${m//:/_}
    timeout $(( $(left) - 3 )) taskset -c "$BW_CPUS" python3 "$S/serve_bitwise.py" "http://$IP:8133/v1" "$m" "$L/$name-bitwise-$ms.json" > "$L/$name-bitwise-$ms.out" 2>&1
    echo "[$name] bitwise $m exit $? $(now): $(head -n 1 "$L/$name-bitwise-$ms.out")"
  done
}
Q=/models/Qwen3.6-35B-A3B-NVFP4
q35() {  # card binary env prompt tokens: greedy `generate` of the 35B (lane ink-hip-dec3's window 6 recipe), md5 of its text
  local card=$1 b=$2 e=$3 p=$4 n=$5
  (( $(left) < 20 )) && { echo "Q35 $b $e $p $n SKIP ($(left) s left)"; return; }
  local PR; PR="$(cat "$REPO/scripts/cmx/ids-prompts/$p-0.txt")"
  local f=$L/q35-$b-${e:-unset}-$p-$n; f=${f//=/}
  HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$(cname "$card") timeout $(( $(left) - 5 )) bash "$S/run.sh" env $e /work/.scratch/bin/intarsia-$b generate -m $Q \
    --backend hip --max-seq 8192 --mem-budget 29 --spec off --ignore-eos --profile-every 0 --max-tokens "$n" -p "$PR" > "$f.out" 2> "$f.err"
  local rc=$?
  local h; h=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "Q35 card$card $b ${e:-unset} $p $n $h rc $rc | $(cat "$f.out" "$f.err" | grep -aoE 'prompt=[0-9]+|decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -2 | tr '\n' ' ') | $(cat "$f.out" "$f.err" | grep -aoE 'steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | head -1) $(grep -aE 'Error|panicked|overlap' "$f.out" "$f.err" | head -2 | cut -c1-200) [$(now)]"
}
