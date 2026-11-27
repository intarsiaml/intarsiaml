#!/usr/bin/env bash
# ON tatooine inside `fleetq run --host tatooine-r9700 --load net,disk`: the DFlash2 drafter for Inkling-2-Mini, pinned to one HF
# revision, into its own new directory beside the target. model.safetensors is checked against the HF LFS sha256. The Python file
# is third-party reference code: downloaded to read, never run.
set -euo pipefail
REV=4f9088ea66c21f2b99ed657d1941c306b03e84d2
SHA=217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee
D=/data/models/intarsia/Inkling-2-Mini-DFlash2
mkdir -p "$D"
cd "$D"
for f in config.json README.md sglang_inkling_dflash2.py model.safetensors; do
  curl -sSfL --retry 3 -o "$f.part" "https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/resolve/$REV/$f"
  mv "$f.part" "$f"
  echo "got $f $(stat -c %s "$f") bytes"
done
got=$(sha256sum model.safetensors | cut -d' ' -f1)
echo "model.safetensors sha256 $got"
[ "$got" = "$SHA" ] || { echo "SHA MISMATCH (want $SHA)"; exit 2; }
echo "$REV" > REVISION
chmod a-w config.json README.md sglang_inkling_dflash2.py model.safetensors
ls -l "$D"
