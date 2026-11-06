#!/bin/zsh
# Lane lmhead-fp8: rebuild the Metal release binary of this worktree in place (cargo serialises concurrent builds). Called at the top of the
# endor job scripts so a job always runs the code at the branch head.
R=${0:A:h:h:h}
cd $R && INTARSIA_BUILD_STAMP="lmhead-head $(git rev-parse --short=9 HEAD)" cargo build --release -p intarsia-cli --features metal 2>&1 | tail -2
echo "rebuild head $(git rev-parse --short=9 HEAD): $(ls -la target/release/intarsia | awk '{print $6,$7,$8}')"
