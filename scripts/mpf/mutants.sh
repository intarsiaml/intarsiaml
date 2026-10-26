#!/usr/bin/env bash
# Lane mpf: every mutant of mutate.py must make its tests fail (TEST, default mtp; the default-mechanism mutants: TEST=tp_mtp).
# Names starting cuda- are CUDA-binary mutants (scripts/mpf/spark1.sh mutant), not run here unless named.
# Run from the lane worktree (cpu only, local):
#   [TEST=tp_mtp] scripts/dfh/mutants.sh [name ...]       (default: all)
# A test binary that runs longer than TMO s (default 600: a deadlocked rank) counts as failing (CAUGHT, timeout).
# The mutant tree is a detached worktree at target/mut (reset between mutants); it prints the sha256 of each mutant's test
# binary against the clean build's, and exits non-zero if a mutant does not apply, does not change the binary, or survives.
set -uo pipefail
REPO=$(git rev-parse --show-toplevel)
MUT=$REPO/target/mut
export CARGO_TARGET_DIR=$REPO/target/mut-target
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST=${TEST:-mtp}; TMO=${TMO:-600}
names=("$@"); [[ ${#names[@]} -eq 0 ]] && names=($(python3 -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('m','$S/mutate.py'); sys.argv=['x','',''];" 2>/dev/null; grep -oE '^    "[a-z0-9-]+":' "$S/mutate.py" | tr -d ' ":' | grep -v '^cuda-'))
[[ -d $MUT ]] || git -C "$REPO" worktree add -q --detach "$MUT" HEAD
git -C "$MUT" checkout -q --detach "$(git -C "$REPO" rev-parse HEAD)" && git -C "$MUT" reset -q --hard
bin() { (cd "$MUT" && cargo test -q --release -p intarsia-models --test "$TEST" --no-run --message-format=json 2>/dev/null) |
  python3 -c "import sys,json; [print(m['executable']) for m in map(json.loads, sys.stdin) if m.get('reason')=='compiler-artifact' and m.get('executable')]" | tail -1; }
clean=$(bin); [[ -n $clean ]] || { echo "clean build failed"; exit 1; }
clean_sha=$(shasum -a 256 "$clean" | cut -c1-16)
echo "clean ($TEST) $(git -C "$MUT" rev-parse --short HEAD) test binary sha256 $clean_sha"
(cd "$MUT" && perl -e "alarm $TMO; exec @ARGV" "$clean" -q > /dev/null 2>&1) || { echo "clean tree FAILS its own tests"; exit 1; }
rc=0
for n in "${names[@]}"; do
  git -C "$MUT" reset -q --hard
  python3 "$S/mutate.py" "$MUT" "$n" || { rc=1; continue; }
  b=$(bin); [[ -n $b ]] || { echo "mutant $n: build failed"; rc=1; continue; }
  sha=$(shasum -a 256 "$b" | cut -c1-16)
  [[ $sha == "$clean_sha" ]] && { echo "mutant $n: binary identical to clean ($sha): NOT A MUTANT"; rc=1; continue; }
  out=$(cd "$MUT" && perl -e "alarm $TMO; exec @ARGV" "$b" 2>&1); r=$?
  [[ $r -eq 142 ]] && out="$out
    timeout_${TMO}s"
  failed=$(echo "$out" | grep -E '^    [a-z0-9_]+$' | tr '\n' ' ')
  if [[ $r -ne 0 ]]; then echo "mutant $n: sha256 $sha (clean $clean_sha): CAUGHT by: $failed"; else echo "mutant $n: sha256 $sha: SURVIVED"; rc=1; fi
done
git -C "$MUT" reset -q --hard
exit $rc
