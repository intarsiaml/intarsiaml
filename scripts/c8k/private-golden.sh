#!/usr/bin/env bash
# Lane c8k, Mac side, inside a fleetq slot on the host (--load none): give the lane its own golden directory on the Spark and take the
# lane's files back out of the shared ~/intarsia-decfuse/pf-golden. The reuse gate finds oracles in the golden's own directory, so an
# oracle for a split that main has no oracle for (here s6000 of the 12k prompts) changes what any other lane's reuse run reports.
# Usage: private-golden.sh <host>
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
H=$1; D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; G="$D/../../golden"
ssh -n -o BatchMode=yes "$H" "mkdir -p ~/intarsia-decfuse/pf-golden-c8k"
rsync -aL --include='qwen36*' --include='oracle/***' --include='ref_noise/***' --include='baselines/***' --include='reuse-splits/***' --exclude='*' "$G/" "$H:intarsia-decfuse/pf-golden-c8k/"
# the lane's files in the shared directory: the oracles that are not on main (12k, long_code_4k -g4k) and the derived golden
ssh -n -o BatchMode=yes "$H" 'cd ~/intarsia-decfuse/pf-golden && rm -f oracle/*long_docs_12k-s*.continue-oracle.safetensors* oracle/*long_code_12k-s*.continue-oracle.safetensors* oracle/*long_code_4k-s2000-g4k.continue-oracle.safetensors* oracle/*long_code_4k-s3072-g4k.continue-oracle.safetensors* && rm -rf reuse-splits && ls oracle | sed "s/qwen36-35b-a3b-nvfp4-continue-//" | tr "\n" " "; echo; ls ../pf-golden-c8k/oracle | wc -l'
