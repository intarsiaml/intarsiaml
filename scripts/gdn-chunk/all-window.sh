#!/usr/bin/env bash
# Lane gdn-chunk, one window with every on-box gate, in one out dir: the kernel tests, the timing sweep, then the accuracy
# steps of $STEPS (accuracy-window.sh).
d="$(dirname "$0")"; "$d/kernels-window.sh"; "$d/timing-window.sh"; "$d/accuracy-window.sh"
