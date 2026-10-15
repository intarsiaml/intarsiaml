#!/usr/bin/env python3
# Timestamp each stdin line with epoch seconds (ms) as it arrives; passes the line through to stdout too.
import sys, time
for line in sys.stdin:
    sys.stdout.write(f"{time.time():.3f}\t{line}"); sys.stdout.flush()
