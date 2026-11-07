#!/usr/bin/env bash
# Build a sibling model dir that serves a tuned MTP head: every file of the original checkpoint is symlinked (dotfiles too), the
# tuned tensors are added as mtp-tuned.safetensors, and model.safetensors.index.json (a COPY, edited) maps the mtp.* names to it.
#
#   make_override.sh <orig-model-dir> <mtp-tuned.safetensors> <new-dir>
#
# The loader (crates/intarsia-models/src/safetensors.rs:36-75) opens the files named in weight_map in SORTED FILENAME order and
# `parse_header` inserts every tensor into one map, so on a duplicate name the later-sorted file wins: "mtp-tuned.safetensors"
# sorts after "model-0000N-of-0000M.safetensors" ('t' > 'o'), so the tuned mtp.* shadow the originals in model-00003. Before
# linking, every tuned tensor's name, dtype and shape is checked against the original's header.
set -euo pipefail
[[ $# == 3 ]] || { sed -n '2,11p' "$0"; exit 2; }
ORIG=$(realpath "$1"); TUNED=$(realpath "$2"); NEW=$3
[[ -d $ORIG ]] || { echo "not a directory: $ORIG" >&2; exit 1; }
[[ -f $TUNED ]] || { echo "not a file: $TUNED" >&2; exit 1; }
[[ -e $NEW && -n $(ls -A "$NEW" 2>/dev/null) ]] && { echo "$NEW exists and is not empty" >&2; exit 1; }
[[ -e $ORIG/mtp-tuned.safetensors ]] && { echo "$ORIG already has an mtp-tuned.safetensors (override of an override?)" >&2; exit 1; }
mkdir -p "$NEW"
trap 'rm -rf "$NEW"' ERR   # no half-built directory after a failed check
shopt -s dotglob nullglob
for f in "$ORIG"/*; do
  b=$(basename "$f")
  [[ $b == model.safetensors.index.json ]] && continue
  ln -s "$(realpath "$f")" "$NEW/$b"
done
ln -s "$TUNED" "$NEW/mtp-tuned.safetensors"
python3 - "$ORIG" "$TUNED" "$NEW" <<'PY'
import json, os, struct, sys
orig, tuned, new = sys.argv[1:4]
def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h
idx = json.load(open(os.path.join(orig, "model.safetensors.index.json")))
wm = idx["weight_map"]
th = header(tuned)
cache = {}
for name, e in th.items():
    if name not in wm:
        sys.exit(f"{name}: not in the original's weight_map")
    f = wm[name]
    cache.setdefault(f, header(os.path.join(orig, f)))
    o = cache[f][name]
    if o["dtype"] != e["dtype"] or o["shape"] != e["shape"]:
        sys.exit(f"{name}: tuned {e['dtype']} {e['shape']} != original {o['dtype']} {o['shape']}")
    wm[name] = "mtp-tuned.safetensors"
missing = [n for n in wm if n.startswith("mtp.") and wm[n] != "mtp-tuned.safetensors"]
if missing:
    print("note: mtp.* tensors not overridden:", missing)
json.dump(idx, open(os.path.join(new, "model.safetensors.index.json"), "w"), indent=2)
print(f"override: {len(th)} tensors -> mtp-tuned.safetensors; {sum(1 for n in wm if n.startswith('mtp.'))} mtp.* names in the index")
PY
echo "wrote $NEW"
