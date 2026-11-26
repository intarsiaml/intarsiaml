#!/usr/bin/env bash
# ON tatooine, inside a fleetq --load disk job: remove the merged lanes' leftovers named in the ink-hip-dec2 brief (only these).
set -uo pipefail
df -h /
for d in intarsiaml-ink-hip-mix intarsiaml-ink-hip-attn intarsiaml-ink-hip-speed intarsiaml-ink-hip; do
  if [ -d "$HOME/$d" ]; then du -sh "$HOME/$d" 2>/dev/null; rm -rf "${HOME:?}/$d" && echo "removed ~/$d"; fi
done
for v in intarsia-cargo-target-ink-hip-mix intarsia-cargo-target-ink-hip-attn intarsia-cargo-target-ink-hip-speed; do
  if docker volume inspect "$v" >/dev/null 2>&1; then docker volume rm "$v" && echo "removed volume $v"; fi
done
df -h /
