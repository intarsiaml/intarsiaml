#!/usr/bin/env python3
import sys,time
for l in sys.stdin:
    sys.stdout.write(f"{time.time():.3f} {l}"); sys.stdout.flush()
