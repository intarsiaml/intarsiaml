"""Window 10 pre-stage verdict (lane ink-hip-batch): PASS only when every unit test that ran passed (ink_rows: both row-table tests;
rows_exact_real_shapes; the tiny inkling2_batch suite), both admissions say ADMITTED, and main and branch give the same greedy md5
for every (prompt, tokens) pair (4k / short at 256 and 64).  python3 gate10_check.py PRE_OUT; exit 0 = PASS."""
import collections
import re
import sys

t = open(sys.argv[1], errors="replace").read()
bad = []
res = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed", t)
if len(res) < 3:
    bad.append(f"only {len(res)} test results (want 3: ink_rows, ink_attn rows_exact, inkling2_batch)")
for r in res:
    if r[0] != "ok" or r[2] != "0" or r[1] == "0":
        bad.append(f"test result {r}")
if re.search(r"panicked|FAILED", t):
    bad.append("panicked / FAILED in the output")
adm = len(re.findall(r"ADMITTED", t))
if adm < 2:
    bad.append(f"{adm} ADMITTED lines (want 2)")
if re.search(r"REJECT|NO CASES", t):
    bad.append("REJECT / NO CASES in an admission")
md5 = collections.defaultdict(dict)
for m in re.finditer(r"SPD (main|branch) (\w+) (\d+) r\d+ ([0-9a-f]{32})", t):
    md5[(m.group(2), m.group(3))][m.group(1)] = m.group(4)
for want in (("4k", "256"), ("short", "256"), ("short", "64"), ("4k", "64")):
    d = md5.get(want, {})
    if len(d) != 2:
        bad.append(f"md5 {want}: missing an arm ({d})")
    elif d["main"] != d["branch"]:
        bad.append(f"md5 {want}: main {d['main'][:12]} != branch {d['branch'][:12]}")
    else:
        print(f"md5 {want[0]} {want[1]}: EQUAL {d['main']}")
print(f"tests {res}, ADMITTED x{adm}")
print("GATE " + ("PASS" if not bad else "FAIL: " + "; ".join(bad)))
sys.exit(0 if not bad else 1)
