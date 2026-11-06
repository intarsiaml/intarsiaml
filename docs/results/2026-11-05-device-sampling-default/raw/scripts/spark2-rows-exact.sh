#!/usr/bin/env bash
# Lane lane-ds: retry of the one workspace-test failure on spark2 (rows_exact SIGSEGV, a CUDA backend kernel test) in a GPU-quiet slot.
ssh -n spark2 'cd ~/intarsiaml-ds/src && export CARGO_BUILD_JOBS=8 && ~/.cargo/bin/cargo test -p intarsia-backend-cuda --test rows_exact 2>&1 | tail -25'
