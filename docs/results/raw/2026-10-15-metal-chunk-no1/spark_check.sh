#!/bin/zsh
# build/CPU-only check of the cuda and hip feature builds on spark1, in a scratch dir that is deleted afterwards
set -u
D=chunk-no1-check-$$
cd /opt/code/intarsiaml-chunk-no1 || exit 1
ssh spark1 "mkdir -p ~/$D" || exit 1
trap 'ssh spark1 "rm -rf ~/'$D'"; echo "scratch ~/'$D' removed"' EXIT
git ls-files -z | rsync -a --from0 --files-from=- ./ spark1:$D/ || exit 1
for f in cuda hip; do
  echo "=== cargo check -p intarsia-cli --features $f --tests"
  ssh spark1 "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; cd ~/$D && nice cargo check -p intarsia-cli --features $f --tests -j 6 2>&1 | tail -25; echo EXIT=\${PIPESTATUS[0]}"
done
