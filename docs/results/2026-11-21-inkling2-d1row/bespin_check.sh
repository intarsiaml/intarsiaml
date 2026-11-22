#!/bin/sh
# lane ink-d1row: cargo check of the Metal crate on bespin (coordinator's conditions, results doc brief 3). usage: bespin_check.sh HASH
H=$1
cd /opt/code/intarsiaml/.worktrees/ink-d1row || exit 1
os=$(ssh -o BatchMode=yes bespin 'uname -s' < /dev/null)
echo "bespin uname -s: $os"
[ "$os" = Darwin ] || { echo "not Darwin: stop"; exit 2; }
ssh -o BatchMode=yes bespin 'xcode-select -p 2>&1; cc --version 2>&1 | head -1' < /dev/null
ssh -o BatchMode=yes bespin 'test -x ~/.cargo/bin/rustup || (curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path --default-toolchain none) > ~/rustup-install.log 2>&1; ~/.cargo/bin/rustup toolchain install 1.90.0 --profile minimal > ~/rustup-tc.log 2>&1; e=$?; tail -2 ~/rustup-tc.log; exit $e' < /dev/null || { echo "rustup failed"; exit 3; }
git archive --format=tar "$H" Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml crates kernels | ssh -o BatchMode=yes bespin "rm -rf ~/intarsiaml-ink-d1row/src && mkdir -p ~/intarsiaml-ink-d1row/src && tar -x -C ~/intarsiaml-ink-d1row/src && echo $H > ~/intarsiaml-ink-d1row/src/HASH" || { echo "copy failed"; exit 4; }
ssh -o BatchMode=yes bespin 'cd ~/intarsiaml-ink-d1row/src && ~/.cargo/bin/rustc --version && CARGO_BUILD_JOBS=4 ~/.cargo/bin/cargo check --release -p intarsia-backend-metal --examples --tests > ~/intarsiaml-ink-d1row/check.log 2>&1; e=$?; grep -E "^(error|warning: unused)" -A12 ~/intarsiaml-ink-d1row/check.log | head -150; tail -5 ~/intarsiaml-ink-d1row/check.log; echo "cargo check exit $e"; exit $e' < /dev/null
