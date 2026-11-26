#!/usr/bin/env python3
"""The lm_head evidence of a rocprofv3 kernel trace: for the last full decode step (between the last two argmax dispatches),
every dispatch of >= 200 us and the 4 dispatches before argmax, with the idle gap before each. usage: tailtrace.py trace.csv"""
import csv, re, sys
rows = sorted(csv.DictReader(open(sys.argv[1])), key=lambda r: int(r["Start_Timestamp"]))
nm = lambda n: re.sub(r"^void ", "", re.sub(r"\(.*$", "", n.replace("(anonymous namespace)::", ""))).replace(" [clone .kd]", "")[:48]
marks = [i for i, r in enumerate(rows) if "argmax" in r["Kernel_Name"]]
if len(marks) < 2:
    sys.exit(f"only {len(marks)} argmax dispatches")
lo, hi = marks[-2] + 1, marks[-1] + 1
for i in range(lo, hi):
    r, p = rows[i], rows[i - 1]
    d = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
    g = (int(r["Start_Timestamp"]) - int(p["End_Timestamp"])) / 1e3
    if d >= 200 or i >= hi - 4:
        print(f"  {nm(r['Kernel_Name']):48s} gap {g:8.2f} us  dur {d:8.2f} us  grid {r['Grid_Size_X']}")
