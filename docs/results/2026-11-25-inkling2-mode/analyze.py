#!/usr/bin/env python3
"""Lane ink-mode: per-process table and fast/slow correlations from a mode.sh run directory.
usage: analyze.py RUN_DIR [CUT_MS]   (CUT_MS: 16k prefill at or above it counts as slow; default: the largest gap)"""
import glob, json, os, re, statistics as st, sys
from datetime import datetime

d = sys.argv[1]
cut = float(sys.argv[2]) if len(sys.argv) > 2 else None

mm = []
if os.path.exists(f"{d}/macmon.jsonl"):
    for l in open(f"{d}/macmon.jsonl"):
        try:
            j = json.loads(l)
        except Exception:
            continue
        t = datetime.fromisoformat(j["timestamp"].replace("Z", "+00:00")).timestamp()
        mm.append((t, j["gpu_usage"][0], j["gpu_usage"][1], j["gpu_power"], j["temp"]["gpu_temp_avg"], j.get("ram_power", 0), j["memory"]["swap_usage"] / 1e9))

VMK = {"Pages free": "free", "Pages stored in compressor": "cmp_stored", "Compressions": "compr", "Decompressions": "decompr",
       "Swapins": "swapin", "Swapouts": "swapout", "Pageins": "pagein", "Pages wired down": "wired", "File-backed pages": "filebacked"}


def vm(f):
    r = {}
    for l in open(f):
        m = re.match(r'"?([^:"]+)"?:\s+(\d+)', l.strip())
        if m and m.group(1) in VMK:
            r[VMK[m.group(1)]] = int(m.group(2))
    return r


rows = []
for err in sorted(glob.glob(f"{d}/*.err")):
    base = err[:-4]
    tag, arm = os.path.basename(base).rsplit("-", 1)
    t = open(err).read()
    pf = {int(a): int(b) for a, b in re.findall(r"^prefill\[(\d)\]: \d+ tokens, (\d+) ms", t, re.M)}
    if 2 not in pf:
        continue
    r = {"tag": tag, "arm": arm, "p1k": pf.get(0), "p4k": pf.get(1), "p16k": pf[2]}
    dec = re.findall(r"^decode\[(\d)\]: \d+ tokens, \d+ ms \(([0-9.]+) tok/s\)", t, re.M)
    for i, v in dec:
        r[f"dec{i}"] = float(v)
    ids = re.findall(r"^ids\[(\d)\]: (.*)$", t, re.M)
    r["ids"] = "|".join(v for _, v in ids)
    for m in re.finditer(r"^modeprobe (end|load) (.*)$", t, re.M):
        site = m.group(1)
        for k, v in re.findall(r"(\w+)=([^ ]+)", m.group(2)):
            if k == "sustain":
                sv = [float(x) for x in v.split(",")]
                r[f"{site}.sus_first"], r[f"{site}.sus_last"], r[f"{site}.sus_ratio"] = sv[0], sv[-1], sv[-1] / sv[0]
                continue
            if "/" in v:
                v = v.split("(")[0].split("/")[0]
            try:
                r[f"{site}.{k}"] = float(v)
            except ValueError:
                pass
    fl = [l for l in t.splitlines() if l.startswith("modeprobe file")]
    for l in fl:
        o = re.search(r"order=(\d+)", l).group(1)
        for k in ("len", "m2m", "m32m", "m1g"):
            pass
        va = re.search(r"va=(0x[0-9a-f]+) m16k=\d+ m2m=(0x[0-9a-f]+) m32m=(0x[0-9a-f]+) m1g=(0x[0-9a-f]+)", l)
        gp = re.search(r"gpu=(0x[0-9a-f]+) m16k=\d+ m2m=(0x[0-9a-f]+) m32m=(0x[0-9a-f]+) m1g=(0x[0-9a-f]+)", l)
        ln = int(re.search(r"len=(\d+)", l).group(1))
        if ln > 10e9:  # the 19.5 GB file
            r["va_m2m"], r["va_m32m"], r["va_m1g"] = (int(x, 16) for x in va.groups()[1:])
            r["gpu_m2m"], r["gpu_m32m"], r["gpu_m1g"] = (int(x, 16) for x in gp.groups()[1:])
            r["va"], r["gpu"] = int(va.group(1), 16), int(gp.group(1), 16)
            r["file_order"] = int(o)
    try:
        fo = json.loads(open(base + ".foreign").read())
        r["foreign_ms"] = sum(fo.values())
        r["foreign_top"] = ",".join(f"{k.split(' ', 1)[1][:14]}:{v:.0f}" for k, v in list(fo.items())[:3])
    except Exception:
        pass
    try:
        a, b = vm(base + ".vm0"), vm(base + ".vm1")
        for k in ("compr", "decompr", "swapin", "swapout", "pagein"):
            r[f"d_{k}"] = b[k] - a[k]
        r["free_gb0"] = a["free"] * 16384 / 1e9
        r["cmp_gb0"] = a["cmp_stored"] * 16384 / 1e9
        r["wired_gb0"] = a["wired"] * 16384 / 1e9
        sy = open(base + ".sys0").read()
        m = re.search(r"used = ([0-9.]+)M", sy)
        r["swap_gb0"] = float(m.group(1)) / 1e3 if m else None
        r["therm_warn"] = 0 if ("No thermal warning" in sy or "CPU_Speed_Limit" not in sy) else 1
    except Exception:
        pass
    try:
        t0, t1 = map(float, open(base + ".t").read().split())
        win = [x for x in mm if t0 <= x[0] <= t1]
        pre = [x for x in mm if t0 - 1.5 <= x[0] < t0]
        if win:
            busy = [x for x in win if x[2] > 0.8]
            r["mhz_busy"] = st.median([x[1] for x in busy]) if busy else None
            r["gpu_w"] = st.median([x[3] for x in busy]) if busy else None
            r["temp0"] = win[0][4]
            r["temp_max"] = max(x[4] for x in win)
        if pre:
            r["idle_util_pre"] = st.mean(x[2] for x in pre)
        r["wall_s"] = t1 - t0
    except Exception:
        pass
    rows.append(r)

if not rows:
    sys.exit("no rows")
p16 = sorted(r["p16k"] for r in rows if r["tag"] not in ("warm", "warmN"))
if cut is None and len(p16) > 3:
    gaps = [(p16[i + 1] - p16[i], (p16[i + 1] + p16[i]) / 2) for i in range(len(p16) - 1)]
    cut = max(gaps)[1]
print(f"16k cut {cut:.0f} ms; sorted 16k: {p16}")
cols = ["tag", "arm", "p1k", "p4k", "p16k", "dec2", "end.alu", "end.seq_file_gbs", "end.seq_anon", "end.gather_file", "end.gather_file2g",
        "end.gather_anon", "end.copy_gbs", "end.f0_res", "va_m32m", "gpu_m32m", "foreign_ms", "mhz_busy", "gpu_w", "temp0", "swap_gb0", "free_gb0",
        "d_compr", "d_swapout", "d_pagein", "wall_s", "foreign_top"]
print("\t".join(c.replace("end.", "") for c in cols))
for r in rows:
    print("\t".join("" if r.get(c) is None else (f"{r[c]:.3g}" if isinstance(r[c], float) else str(r[c])) for c in cols))

# fast vs slow
print("\nfeature\tarm\tfast median (n)\tslow median (n)\tspearman rho vs p16k")


def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0] * len(v)
    for i, j in enumerate(s):
        rk[j] = i
    return rk


def rho(x, y):
    if len(x) < 4:
        return float("nan")
    a, b = rank(x), rank(y)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    return num / den if den else float("nan")


feats = [c for c in cols if c not in ("tag", "arm", "foreign_top", "p16k")] + ["va_m2m", "gpu_m2m", "va_m1g", "gpu_m1g", "file_order", "cmp_gb0", "wired_gb0",
                                                                            "idle_util_pre", "temp_max", "end.sus_first", "end.sus_last", "end.sus_ratio", "load.sus_ratio", "d_decompr", "d_swapin", "load.alu", "dec0", "dec1"]
for arm in sorted(set(r["arm"] for r in rows)) + ["all"]:
    rs = [r for r in rows if r["tag"] not in ("warm", "warmN") and (arm == "all" or r["arm"] == arm)]
    fast = [r for r in rs if r["p16k"] < cut]
    slow = [r for r in rs if r["p16k"] >= cut]
    print(f"-- arm {arm}: {len(fast)} fast, {len(slow)} slow; 16k median fast {st.median([r['p16k'] for r in fast]) if fast else '-'} slow {st.median([r['p16k'] for r in slow]) if slow else '-'}")
    for c in feats:
        xs = [(r[c], r["p16k"]) for r in rs if isinstance(r.get(c), (int, float))]
        if len(xs) < 4:
            continue
        f = [r[c] for r in fast if isinstance(r.get(c), (int, float))]
        s = [r[c] for r in slow if isinstance(r.get(c), (int, float))]
        fm = f"{st.median(f):.4g} ({len(f)})" if f else "-"
        sm = f"{st.median(s):.4g} ({len(s)})" if s else "-"
        print(f"{c}\t{arm}\t{fm}\t{sm}\t{rho([x for x, _ in xs], [y for _, y in xs]):+.2f}")
ids = {}
for r in rows:
    ids.setdefault(r["ids"], []).append(f"{r['tag']}-{r['arm']}")
print(f"\ngreedy id sets: {len(ids)} distinct over {len(rows)} processes" + "".join(f"\n  set {i}: {len(v)} processes, e.g. {v[:4]}" for i, v in enumerate(ids.values())))
