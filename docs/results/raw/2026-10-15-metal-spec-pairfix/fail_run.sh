#!/bin/zsh
cd /opt/code/intarsiaml-spec
cargo test -p intarsia-backend-metal --release --test multirow pair_ 2>&1
