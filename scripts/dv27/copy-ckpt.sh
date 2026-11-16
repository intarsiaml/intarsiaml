#!/usr/bin/env bash
# dv27: copy the 27B CUDA checkpoint (Quark AWQ MXFP4, 18 GB) from the Mac to spark1:~/models, resumable, then compare sizes + sha256 of the small files and sizes of the big one.
set -euo pipefail
SRC=/Users/josef/Models/Qwen3.8-27B-Quark-AWQ-MXFP4
DST=spark1:models/
for i in 1 2 3; do rsync -a --partial --progress "$SRC" "$DST" && break; echo "rsync attempt $i failed"; sleep 5; done
(cd "$SRC" && find . -type f -exec stat -f '%z %N' {} + | sort -k2) > /tmp/dv27-src.sizes
ssh spark1 'cd ~/models/Qwen3.8-27B-Quark-AWQ-MXFP4 && find . -type f -exec stat -c "%s %n" {} + | sort -k2' > /tmp/dv27-dst.sizes
diff /tmp/dv27-src.sizes /tmp/dv27-dst.sizes && echo "sizes identical"
a=$(shasum -a 256 "$SRC/model.safetensors" | cut -d' ' -f1)
b=$(ssh spark1 'sha256sum ~/models/Qwen3.8-27B-Quark-AWQ-MXFP4/model.safetensors' | cut -d' ' -f1)
echo "sha256 src $a dst $b"; [ "$a" = "$b" ] && echo "sha256 identical"
