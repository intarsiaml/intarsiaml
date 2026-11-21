#!/usr/bin/env bash
# Lane rocm102 (ON tatooine, CPU only): ARMS="r7 r10" (default both). Per arm: the stamped release CLI; for r10 also the HIP /
# models test binaries (the correctness window runs them on 10.2 only; r7 is main's tested toolchain).
set -u
. "$(dirname "$0")/lib.sh"
for a in ${ARMS:-r7 r10}; do
  arm "$a"
  t0=$(date +%s)
  "$D" env INTARSIA_BUILD_STAMP="$STAMP" cargo build --release --locked -p intarsia-cli --features hip < /dev/null > "$O/build-$a.log" 2>&1
  echo "BUILD $a cli exit $? ($(( $(date +%s) - t0 )) s)"; grep -aE '^(warning|error)(\[|:)' "$O/build-$a.log" | sort | uniq -c | sort -rn | head -8
  if [[ $a == r10 ]]; then
    "$D" cargo test --release --locked -p intarsia-backend-hip --tests --no-run < /dev/null >> "$O/build-$a.log" 2>&1; echo "BUILD $a hip tests exit $?"
    "$D" cargo test --release --locked -p intarsia-models --test batch_recurrent --test mtp --test batch_cuda_real --no-run < /dev/null >> "$O/build-$a.log" 2>&1; echo "BUILD $a models tests exit $?"
  fi
done
echo "logs: $O"
