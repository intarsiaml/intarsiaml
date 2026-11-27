#!/bin/bash
set -eu
D=/Users/josef/.cache/huggingface/hub/models--incoai--Inkling-2-Mini-DFlash2/snapshots/4f9088ea66c21f2b99ed657d1941c306b03e84d2
mkdir -p $D
scp -q tatooine:/data/models/intarsia/Inkling-2-Mini-DFlash2/{config.json,model.safetensors,README.md,REVISION} $D/
ls -l $D; md5 -q $D/model.safetensors; shasum -a 256 $D/model.safetensors
