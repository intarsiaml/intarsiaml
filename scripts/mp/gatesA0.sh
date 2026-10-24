#!/usr/bin/env bash
# Lane mp: control KL run on the main build (no acc-short: that build is not stamped). See gatesA.sh.
SKIP_ACC=1 exec bash "$(dirname "$0")/gatesA.sh" "$@"
