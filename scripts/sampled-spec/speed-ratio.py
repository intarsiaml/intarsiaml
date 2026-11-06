"""Per-round speedups over the spec-off arm at the same temperature, averaged over rounds (rounds drift by up to ~20 % in
absolute tok/s on endor, so absolute medians are not pooled): speed-ratio.py <round dir>...
Prints, per model and arm, x off at 1k and 4k per round, the mean over the 2 sizes x rounds, and the share of the greedy
(T0 MTP) mean gain that survives: (mean - 1) / (greedy mean - 1)."""
import collections, glob, os, re, statistics, sys

ARMS = ["T00-mtp", "T07-match", "T07-reject", "T10-match", "T10-reject"]
ratios = collections.defaultdict(list)
cells = collections.defaultdict(list)
for d in sys.argv[1:]:
    med = {}
    for f in glob.glob(os.path.join(d, "*.localeval.log")):
        tag, arm = os.path.basename(f)[: -len(".localeval.log")].split("-", 1)
        v = collections.defaultdict(list)
        for m in re.finditer(r"\[localeval\]\s+(\d+) r\d+ p\d+: .*decode=([\d.]+)", open(f).read()):
            v[int(m.group(1))].append(float(m.group(2)))
        for sz, xs in v.items():
            med[(tag, arm, sz)] = statistics.median(xs)
    for (tag, arm, sz), x in med.items():
        if arm in ARMS:
            off = med.get((tag, arm.split("-")[0] + "-off", sz))
            if off:
                ratios[(tag, arm)].append(x / off)
                cells[(tag, arm)].append(f"{x / off:.3f}")
print("| model | arm | x off per round and size (1k, 4k, ...) | mean x off | greedy gain kept |")
print("|---|---|---|---:|---:|")
for tag in sorted({k[0] for k in ratios}):
    g = statistics.mean(ratios[(tag, "T00-mtp")])
    for arm in ARMS:
        r = ratios.get((tag, arm))
        if not r:
            continue
        m = statistics.mean(r)
        kept = "100 %" if arm == "T00-mtp" else f"{(m - 1) / (g - 1) * 100:.0f} %"
        print(f"| {tag} | {arm} | {', '.join(cells[(tag, arm)])} | {m:.3f} | {kept} |")
