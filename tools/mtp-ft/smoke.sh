#!/usr/bin/env bash
# End-to-end smoke test on synthetic dumps with the REAL mtp.* / embed / lm_head weights (run inside the pytorch container on a GPU
# box, see README.md): head forward, dump fixture, parity, accept, a few training steps, saved-file name/dtype/shape check,
# override dir. Aborts when host MemAvailable < ${MIN_AVAIL_GB:-15} GB (a GB10 hangs instead of OOM-killing).
#   bash smoke.sh <checkpoint dir> <scratch dir>
set -euo pipefail
CK=${1:?checkpoint dir}; WD=${2:?scratch dir}
cd "$(dirname "$0")"
mkdir -p "$WD"
( while sleep 4; do
    a=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
    if (( a < ${MIN_AVAIL_GB:-15} )); then echo "MemAvailable ${a} GB: killing python" >&2; pkill -9 python || true; exit; fi
  done ) &
WATCH=$!; trap 'kill $WATCH 2>/dev/null || true' EXIT
python selftest.py
python - <<PY
import time, torch, head as H
t = time.time(); h = H.load_head("$CK", "cuda"); torch.cuda.synchronize()
print(f"load_head {time.time()-t:.1f}s, GPU {torch.cuda.memory_allocated()/2**30:.1f} GB; lm_head {tuple(h.lm_w.shape)} {h.lm_w.dtype} scale {h.lm_scale}")
names = h.named_mtp(); print(len(names), "mtp tensors, params", sum(p.numel() for p in names.values())/1e6, "M")
PY
python synth.py dumps --ckpt "$CK" --out "$WD/dumps" -n 6 --len 48 --draft-vocab 65536 --device cuda
python parity.py --ckpt "$CK" --dumps "$WD/dumps" | tail -3
python accept.py --ckpt "$CK" --dumps "$WD/dumps" --meta "$WD/dumps/meta.json"
rm -rf "$WD/out"
python train.py --ckpt "$CK" --dumps "$WD/dumps" --meta "$WD/dumps/meta.json" --out "$WD/out" --heldout-pct 30 --tokens-per-batch 2048 \
  --accum 1 --epochs 4 --lr 1e-4 --warmup 2 --eval-every 6 --workers 0 | grep -v '"categories"'
python accept.py --ckpt "$CK" --dumps "$WD/dumps" --meta "$WD/dumps/meta.json" --mtp-file "$WD/out/mtp-tuned.safetensors" | tail -1
rm -rf "$WD/override"
bash make_override.sh "$CK" "$WD/out/mtp-tuned.safetensors" "$WD/override"
ls -la "$WD/override" | head -8
echo SMOKE OK
