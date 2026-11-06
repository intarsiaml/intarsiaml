"""Speed table of the sampled-speculation runs: speed-table.py <dir with <tag>-<arm>.localeval.log / .metrics0 / .metrics1>.

Per model tag and arm: median decode tok/s of the speed-spec reps at 1k and 4k (the `[localeval] SIZE rN pM: ... decode=X` lines),
acceptance over the whole run from the server's /metrics counters (probe and warmup requests included), the gain over the
spec-off arm at the same temperature, and the fraction of the greedy (T0) MTP gain that survives: (gain_T - 1) / (gain_T0 - 1).
"""
import collections, glob, os, re, statistics, sys

d = sys.argv[1]
dec = collections.defaultdict(list)
acc = {}
for f in glob.glob(os.path.join(d, "*.localeval.log")):
    tag, arm = os.path.basename(f)[: -len(".localeval.log")].split("-", 1)
    for m in re.finditer(r"\[localeval\]\s+(\d+) r\d+ p\d+: .*decode=([\d.]+)", open(f).read()):
        dec[(tag, arm, int(m.group(1)))].append(float(m.group(2)))
    run = re.search(r"saved (\S+)/result.json", open(f).read())

    def counters(p):
        c = {}
        if os.path.exists(p):
            for line in open(p):
                k, v = line.split()
                c[k] = float(v)
        return c

    c0 = counters(f.replace(".localeval.log", ".metrics0"))
    c1 = counters(f.replace(".localeval.log", ".metrics1"))
    p = c1.get("intarsia_spec_proposed_tokens_total", 0) - c0.get("intarsia_spec_proposed_tokens_total", 0)
    a = c1.get("intarsia_spec_accepted_tokens_total", 0) - c0.get("intarsia_spec_accepted_tokens_total", 0)
    s = c1.get("intarsia_spec_steps_total", 0) - c0.get("intarsia_spec_steps_total", 0)
    acc[(tag, arm)] = (a / p if p else None, (s + a) / s if s else None, run.group(1).split("/")[-1] if run else "-")

tags = sorted({k[0] for k in dec})
arms = ["T00-off", "T00-mtp", "T07-off", "T07-match", "T07-reject", "T10-off", "T10-match", "T10-reject"]
print("| model | arm | decode 1k | decode 4k | x off 1k | x off 4k | gain kept 1k | gain kept 4k | acceptance | tok/pass | localeval run |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for tag in tags:
    med = {(arm, sz): statistics.median(dec[(tag, arm, sz)]) for arm in arms for sz in (1024, 4096) if dec.get((tag, arm, sz))}
    for arm in arms:
        if (arm, 1024) not in med and (arm, 4096) not in med:
            continue
        t = arm.split("-")[0]
        row = [tag, arm]
        for sz in (1024, 4096):
            row.append(f"{med[(arm, sz)]:.1f}" if (arm, sz) in med else "-")
        for sz in (1024, 4096):
            off = med.get((f"{t}-off", sz))
            row.append(f"{med[(arm, sz)] / off:.3f}" if off and (arm, sz) in med else "-")
        for sz in (1024, 4096):
            off0, mtp0, off = med.get(("T00-off", sz)), med.get(("T00-mtp", sz)), med.get((f"{t}-off", sz))
            if arm.endswith("off") or not (off0 and mtp0 and off and (arm, sz) in med) or mtp0 <= off0:
                row.append("-")
            else:
                row.append(f"{(med[(arm, sz)] / off - 1) / (mtp0 / off0 - 1) * 100:.0f}%")
        a, tp, run = acc.get((tag, arm), (None, None, "-"))
        row += [f"{a:.3f}" if a is not None else "-", f"{tp:.2f}" if tp else "-", run]
        print("| " + " | ".join(row) + " |")
