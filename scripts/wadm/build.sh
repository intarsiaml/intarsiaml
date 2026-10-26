#!/usr/bin/env bash
# Lane wadm, ON tatooine inside a fleetq job (--load cpu), no GPU: build the hip admission test binary of the clean tree and of four mutants
# into a bind-mounted cargo target dir on /data (volume intarsia-cargo-wadm -> /data/intarsiaml-wadm-target), copying each binary to
# /data/intarsiaml-wadm-target/bin/<name>. The tree is ~/intarsiaml-wadm (scripts/wadm/sync.sh). Mutants (each restored before the next):
#   m1  device.rs: matmul_pair_bf16_wmma writes its two outputs swapped
#   m2  device.rs: matmul_fp8_pair_wmma writes its two outputs swapped
#   m3  device.rs: rmsnorm_add_matmul_fp8_pair_wmma writes its first two outputs (sum, normed row) swapped
#   k1  ik_f8_rows.h: the W8A8 WMMA path drops the second half of each 32-wide K chunk (kernel bug, recompiles the kernels)
#   fleetq run --host tatooine --load cpu --expect 40 --label "wadm: build" -- ssh tatooine 'bash -s' < scripts/wadm/build.sh
set -uo pipefail
VOL=intarsia-cargo-wadm; DIR=/data/intarsiaml-wadm-target; T=$HOME/intarsiaml-wadm
mkdir -p "$DIR/bin" && chmod 0777 "$DIR" "$DIR/bin"
docker volume inspect "$VOL" >/dev/null 2>&1 || docker volume create --driver local -o type=none -o o=bind -o device="$DIR" "$VOL" >/dev/null
export INTARSIA_REPO=$T INTARSIA_TARGET_VOL=$VOL
D=$T/deploy/tatooine/dev.sh
build() { # <name>
  echo "== build $1 $(date +%T)"
  $D env CARGO_BUILD_JOBS=20 cargo test --release --locked --no-run -p intarsia-backend-hip --test admission < /dev/null 2>&1 | grep -E '^(error|warning: unused)|Finished|-->' | tail -8
  $D bash -c 'f=$(ls -t /target/release/deps/admission-* | grep -v "\." | head -1); cp "$f" /target/bin/'"$1"'; ls -la /target/bin/'"$1" < /dev/null
}
mutate() { # <file> <sed expr>
  cp "$T/$1" "$T/$1.orig" && sed -i -E "$2" "$T/$1" && ! cmp -s "$T/$1" "$T/$1.orig" || { echo "MUTATION DID NOT APPLY: $1 $2"; exit 1; }
}
restore() { mv "$T/$1.orig" "$T/$1"; }
build clean
F=crates/intarsia-backend-hip/src/device.rs
mutate $F 's/(ik_matmul_pair_bf16_wmma\(i\(0\), i\(1\), i\(2\), )o, o\.add\(1\)/\1o.add(1), o/'; build m1; restore $F
mutate $F 's/(ik_matmul_fp8_pair_wmma\(i\(0\), i\(1\), i\(2\), i\(3\), i\(4\), )o, o\.add\(1\)/\1o.add(1), o/'; build m2; restore $F
# m3: the first two outputs of the fused fp8 pair are the lines `o,` / `o.add(1),` after `i(6),` inside the rmsnorm_add_matmul_fp8_pair_wmma call
python3 - "$T/$F" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read(); open(p + ".orig", "w").write(s)
i = s.index('"rmsnorm_add_matmul_fp8_pair_wmma" =>'); j = s.index("o.add(3),", i)
seg = s[i:j]; new = seg.replace("                    o,\n                    o.add(1),\n", "                    o.add(1),\n                    o,\n")
assert new != seg, "m3 did not apply"
open(p, "w").write(s[:i] + new + s[j:])
PY
build m3; restore $F
H=kernels/hip/ik_f8_rows.h
mutate $H 's/^( +)if \(full\) acc\[t\] = ik_f8_wmma\(wa\.z, wa\.w, b\.z, b\.w, acc\[t\]\);/\1(void)0;/'; build k1; restore $H
ls -la "$DIR/bin"; df -h /data | tail -1; echo BUILD DONE
