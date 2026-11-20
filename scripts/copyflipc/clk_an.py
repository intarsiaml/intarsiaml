#!/usr/bin/env python3
"""usage: clk_an.py DIR : per server window (clk-<arm>-r<N>.log, lines "epoch sm_mhz power_w util_pct temp_c") the median SM clock over busy samples (util >= 50 %), min and count; windows below 2300 MHz are flagged."""
import glob, re, statistics as st, sys
for f in sorted(glob.glob(sys.argv[1] + "/clk-*-r*.log")):
    rows = [l.split() for l in open(f) if len(l.split()) == 5]
    busy = [int(r[1]) for r in rows if r[3].isdigit() and int(r[3]) >= 50 and r[1].isdigit()]
    if not busy: print(f"{f.split('/')[-1]}: no busy samples ({len(rows)} rows)"); continue
    m = st.median(busy)
    print(f"{f.split('/')[-1][4:-4]:8s} busy samples {len(busy):4d}  median {m:.0f} MHz  min {min(busy)}  {'DISCARD (<2300)' if m < 2300 else 'ok'}")
