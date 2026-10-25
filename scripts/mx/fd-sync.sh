#!/usr/bin/env bash
# Lane mx fold-default (from the Mac): put a committed tree on tatooine as ~/intarsiaml-mx-<name> (git archive, exactly <rev>),
# plus the 27B HF goldens from /data/models/intarsia/golden (their sha256 is in golden/*.provenance.json), the hp prompts,
# the tree's `git rev-list` (mx-out/revlist, for `accuracy --revlist`: the tatooine tree has no .git) and, with TAILS=<file>,
# the tails golden (scripts/mx/make_tails.py output) as ~/mx/g27-tails2.safetensors. Prints the build stamp for build.sh.
#   scripts/mx/fd-sync.sh dev HEAD      scripts/mx/fd-sync.sh base origin/main
set -euo pipefail
name=${1:?name}; rev=${2:?rev}
cd "$(git rev-parse --show-toplevel)"
dst=intarsiaml-mx-$name
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "set -e; rm -rf ~/$dst.new && mkdir -p ~/$dst.new && tar -x -C ~/$dst.new && rm -rf ~/$dst && mv ~/$dst.new ~/$dst && mkdir -p ~/$dst/p1-golden ~/$dst/mx-out ~/mx && cp /data/models/intarsia/golden/qwen3.8-27b-mxfp4-hfquark-*top20.safetensors ~/$dst/p1-golden/ && cp ~/hp-prompts/prompt*.txt ~/$dst/mx-out/"
git rev-list "$rev" | ssh -o BatchMode=yes tatooine "cat > ~/$dst/mx-out/revlist"
if [[ -n ${TAILS:-} ]]; then scp -q "$TAILS" tatooine:mx/g27-tails2.safetensors; fi
# the goldens must be the ones the provenance files name
for g in golden/qwen3.8-27b-mxfp4-hfquark*-top20.safetensors.provenance.json; do
  f=$(basename "$g" .provenance.json); want=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$g")
  got=$(ssh -o BatchMode=yes tatooine "sha256sum < ~/$dst/p1-golden/$f | cut -c1-64" < /dev/null)
  [[ $got == "$want" ]] || { echo "GOLDEN SHA MISMATCH $f: $got != $want" >&2; exit 3; }
done
tree=$(git rev-parse "$rev^{tree}")
stamp="$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
echo "$stamp" | ssh -o BatchMode=yes tatooine "cat > ~/$dst/.stamp"
echo "$name=$stamp"
