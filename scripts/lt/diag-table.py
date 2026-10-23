#!/usr/bin/env python3
"""Lane lt: tabulate `accuracy --mode reuse --reuse-diag --reuse-measure` logs (diag.sh) per arm and prompt.
Usage: diag-table.py <diag-*.log> ...  (arm = the file's diag-<arm>-... name part)."""
import re, sys, os
cols = ["arm", "prompt", "gate", "agree", "nu_max", "kl_max", "wc_kl_mean", "wc_kl_max", "wc_flips", "c1024_kl_mean", "c2047_kl_mean", "decodeKV_kl_mean", "div_margins"]
print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
for f in sys.argv[1:]:
    arm = os.path.basename(f).split("-")[1]
    rows, cur = {}, None
    for ln in open(f, errors="replace"):
        m = re.search(r"REUSE-MEASURE (\S+) continue split \d+ shared_rows \d+ nu_max (\S+) kl_max (\S+)", ln)
        if m:
            cur = m.group(1); rows.setdefault(cur, {})
            rows[cur].update(nu_max=m.group(2), kl_max=m.group(3)); continue
        m = re.match(r"(\S+) continue split \d+ turn2 \d+ cached \d+ agree (\d+/\d+)", ln)
        if m:
            cur = m.group(1); rows.setdefault(cur, {})["agree"] = m.group(2); continue
        m = re.match(r"(\S+)\s+continue\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+/\d+\s+(ok|FAIL)", ln)
        if m:
            rows.setdefault(m.group(1), {})["gate"] = m.group(2); continue
        if re.match(r"\S+\s+(restore|aligned)\s", ln) or "REUSE-MEASURE" in ln and "restore" in ln:
            cur = None
        if cur is None:
            continue
        m = re.search(r"KL mean (\S+) max (\S+)", ln); fl = re.search(r"argmax flips\s+(\d+)", ln)
        if "warm vs cold (same context)" in ln and m:
            rows[cur].update(wc_kl_mean=m.group(1), wc_kl_max=m.group(2), wc_flips=fl.group(1))
        elif re.search(r"cold chunk \d+ vs cold chunk (\d+)", ln) and m:
            c2 = int(re.search(r"vs cold chunk (\d+)", ln).group(1)); c1 = int(re.search(r"cold chunk (\d+) vs", ln).group(1))
            if c2 == c1 // 2 or (c1 == 1024 and c2 == 512): rows[cur]["c1024_kl_mean"] = f"{m.group(1)} (vs {c2})"
            elif c2 == c1 - 1: rows[cur]["c2047_kl_mean"] = f"{m.group(1)} (vs {c2})"
        elif "decode-KV only" in ln and m:
            rows[cur]["decodeKV_kl_mean"] = m.group(1)
        elif "first divergence" in ln:
            mm = re.findall(r"= ([+-]\S+);?", ln); rows[cur]["div_margins"] = "/".join(x.rstrip(";") for x in mm)
    for p, r in rows.items():
        print("| " + " | ".join([arm, p] + [r.get(c, "") for c in cols[2:]]) + " |")
