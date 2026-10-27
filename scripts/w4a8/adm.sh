#!/usr/bin/env bash
# Lane W4A8, ON tatooine inside a fleetq job that takes radiance down: the full admission sweep of the MXFP4 matmul kernels
# (single and pair), verbose, on card 0. The trap brings radiance back and waits until it answers.
R=$HOME/intarsiaml-w4a8; V=intarsia-cargo-w4a8; D=$R/deploy/tatooine/dev.sh
trap 'systemctl --user start radiance; for i in $(seq 1 48); do curl -s -m 3 http://localhost:8080/v1/models | grep -q "\"data\":\[" && break; sleep 5; done; echo "radiance $(systemctl --user is-active radiance) after $((i*5)) s"' EXIT
systemctl --user stop radiance; sleep 3
for op in matmul_mxfp4 matmul_mxfp4_pair; do
  echo "== kernels test --op $op (full sweep, -v)"
  INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V HIP_VISIBLE_DEVICES=0 $D /target/release/intarsia kernels test --backend hip --op $op -v < /dev/null 2>&1 \
    | grep -aE "a8|REJECT|FAIL|ADMITTED|mismatch|^matmul" | cut -c1-260 | head -60
done
